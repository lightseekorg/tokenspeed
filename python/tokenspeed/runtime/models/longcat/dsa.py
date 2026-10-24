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

"""LongCat owner/reuse sparse attention on the shared MLA prologue."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from tokenspeed_kernel.ops.attention.dsa import dsa_decode_topk, dsa_prefill_topk
from tokenspeed_kernel.ops.longcat.dsa import longcat_dsa_candidates
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    current_forward_ctx,
    slice_to_real_tokens,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.dcp.indexer import select_dsa_topk
from tokenspeed.runtime.layers.layernorm import FusedRMSNorm, LayerNorm, RMSNorm
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3AttentionMLA
from tokenspeed.runtime.models.glm5 import (
    GlmDsaIndexer,
    GlmDsaIndexerOutput,
    GlmMoeDsaAttention,
    _glm_dsa_rope_scaling,
)
from tokenspeed.runtime.models.longcat.selection import (
    DSADecodeSelection as GlmDsaDecodeTopK,
)
from tokenspeed.runtime.models.longcat.selection import (
    DSAPrefillSelection as GlmDsaPrefillTopK,
)
from tokenspeed.runtime.models.longcat.selection import (
    DSASelection,
)
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.env import global_server_args_dict

_INDEXER_PREFILL_MAX_LOGITS_MB_ARG = "deepseek_v4_indexer_prefill_max_logits_mb"


def _slice_indexer_input_rows(
    hidden_states: torch.Tensor, *, expected_rows: int
) -> torch.Tensor:
    """Align the indexer projection input with the real Q/position rows."""
    actual_rows = int(hidden_states.shape[0])
    if actual_rows < expected_rows:
        raise RuntimeError(
            f"DSA indexer expected {expected_rows} rows, but input only has {actual_rows}."
        )
    if actual_rows == expected_rows:
        return hidden_states
    return hidden_states[:expected_rows]


class LongcatDsaIndexer(GlmDsaIndexer):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        q_lora_rank: int,
        qk_rope_head_dim: int,
        rope_theta: float,
        rope_scaling: dict[str, Any] | None,
        max_position_embeddings: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        index_k_norm_type: str = "layernorm",
        rope_is_neox_style: bool | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.index_topk = config.index_topk
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.rope_head_dim = int(qk_rope_head_dim)
        self.softmax_scale = self.index_head_dim ** (-0.5)
        self.weights_softmax_scale = self.softmax_scale * self.index_n_heads ** (-0.5)
        if self.rope_head_dim <= 0 or self.rope_head_dim > self.index_head_dim:
            raise ValueError(
                f"GLM DSA indexer requires 0 < qk_rope_head_dim <= index_head_dim; got qk_rope_head_dim={self.rope_head_dim}, index_head_dim={self.index_head_dim}"
            )
        self.wq_b = ReplicatedLinear(
            q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.index_head_dim, self.index_n_heads],
            bias=False,
            quant_config=None,
            prefix=add_prefix("wk_weights_proj", prefix),
        )
        self._wk_weights_proj_loaded = False
        if index_k_norm_type == "layernorm":
            self.k_norm = LayerNorm(self.index_head_dim, eps=1e-06)
        elif index_k_norm_type == "rms":
            self.k_norm = RMSNorm(self.index_head_dim, eps=1e-06)
        else:
            raise ValueError(
                f"DSA index_k_norm_type must be 'layernorm' or 'rms', got {index_k_norm_type!r}"
            )
        rope_scaling = _glm_dsa_rope_scaling(rope_scaling)
        self.rotary_emb = get_rope(
            self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=(
                not getattr(config, "indexer_rope_interleave", False)
                if rope_is_neox_style is None
                else rope_is_neox_style
            ),
        )

    def forward(self, hidden_states, q_lora, positions) -> GlmDsaIndexerOutput:
        if not self._wk_weights_proj_loaded:
            raise RuntimeError("DSA indexer packed projection weights were not loaded")
        return super().forward(hidden_states, q_lora, positions)


class LongcatDsaAttention(GlmMoeDsaAttention):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        layer_id=None,
        prefix: str = "",
        reduce_attn_results=True,
        alt_stream: torch.cuda.Stream | None = None,
        skip_rope: bool = False,
        is_nextn: bool = False,
        *,
        q_lora_scale: float | None,
        kv_lora_scale: float | None,
        initial_tokens: int,
        local_tokens: int,
        computes_selection: bool | None,
        selection_owner_layer_id: int | None,
        index_k_norm_type: str,
        indexer_rope_is_neox_style: bool | None,
        lora_norm_eps: float,
        decode_query_length_limit: int,
    ) -> None:
        if computes_selection is None or selection_owner_layer_id is None:
            raise ValueError("LongCat requires an explicit selection owner/reuse pair")
        if not 1 <= decode_query_length_limit <= 8:
            raise ValueError(
                f"DSA decode query length limit must be in 1-8, got {decode_query_length_limit}."
            )
        rope_scaling = _glm_dsa_rope_scaling(rope_scaling)
        DeepseekV3AttentionMLA.__init__(
            self,
            config=config,
            mapping=mapping,
            hidden_size=hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=prefix,
            reduce_attn_results=reduce_attn_results,
            alt_stream=alt_stream,
            skip_rope=skip_rope,
            q_lora_scale=q_lora_scale,
            kv_lora_scale=kv_lora_scale,
        )
        if q_lora_rank is None:
            raise ValueError("GLM DSA requires q_lora_rank.")
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=lora_norm_eps)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=lora_norm_eps)
        self.fused_qk_layernorm = FusedRMSNorm(self.q_a_layernorm, self.kv_a_layernorm)
        self.index_topk = config.index_topk
        self.attn_mqa.dsa_candidate_policy = longcat_dsa_candidates
        self.decode_query_length_limit = decode_query_length_limit
        self.initial_tokens = int(initial_tokens)
        self.local_tokens = int(local_tokens)
        self.is_nextn = is_nextn
        self.uses_explicit_selection = computes_selection is not None
        self.computes_selection = bool(computes_selection)
        self.selection_owner_layer_id = selection_owner_layer_id
        if is_nextn:
            raise ValueError("Explicit DSA selection is not supported for NextN")
        if selection_owner_layer_id is None:
            raise ValueError("Explicit DSA selection requires an owner layer id")
        expected_layer_id = (
            selection_owner_layer_id
            if self.computes_selection
            else selection_owner_layer_id + 1
        )
        if layer_id != expected_layer_id:
            role = "owner" if self.computes_selection else "reuse"
            raise ValueError(
                f"DSA {role} layer id must be {expected_layer_id}, got {layer_id}"
            )
        self._forward_selection = (
            DSASelection(owner_layer_id=selection_owner_layer_id)
            if self.computes_selection
            else None
        )
        self.skip_indexer_topk = not self.computes_selection
        if self.skip_indexer_topk and (not self.is_nextn):
            self.indexer = None
        else:
            self.indexer = LongcatDsaIndexer(
                config=config,
                hidden_size=hidden_size,
                q_lora_rank=q_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                prefix=add_prefix("indexer", prefix),
                index_k_norm_type=index_k_norm_type,
                rope_is_neox_style=indexer_rope_is_neox_style,
            )
        self._decode_topk_indices_buffer: torch.Tensor | None = None
        self._decode_topk_lens_buffer: torch.Tensor | None = None
        self._retired_decode_workspaces: list[torch.Tensor] = []

    def _compute_indexer_selections(
        self,
        *,
        hidden_states: torch.Tensor,
        q_norm: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        comm_manager: CommManager,
        num_attn_tokens: int,
        num_prefill_tokens: int,
        prefill_selection: GlmDsaPrefillTopK | None,
        decode_selection: GlmDsaDecodeTopK | None,
    ) -> tuple[GlmDsaPrefillTopK | None, GlmDsaDecodeTopK | None]:
        hidden_states = comm_manager.pre_attn_comm(hidden_states, ctx)
        hidden_states = _slice_indexer_input_rows(
            hidden_states, expected_rows=num_attn_tokens
        )
        indexer_output = self.indexer(hidden_states, q_norm, positions)
        if indexer_output.key is None:
            raise RuntimeError("unprepared DSA Indexer output must contain index K")
        ctx.token_to_kv_pool.set_index_k_buffer(
            self.attn_mqa.layer_id,
            out_cache_loc,
            indexer_output.key,
            write_mask=None,
        )
        if ctx.num_extends > 0:
            prefill_selection = self._compute_prefill_topk_indices(
                indexer_output, ctx, num_prefill_tokens
            )
        if ctx.num_extends < ctx.bs:
            decode_selection = self._compute_decode_topk_indices(indexer_output, ctx)
        self._forward_selection.prefill = prefill_selection
        self._forward_selection.decode = decode_selection
        return (prefill_selection, decode_selection)

    def forward_with_selection(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        comm_manager: CommManager,
        *,
        selection: DSASelection | None,
        block_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DSASelection]:
        """Run an explicit owner/reuse attention and return its pair-local selection."""
        out_cache_loc = ctx.attn_backend.padded_write_locations(
            self.attn_mqa, ctx.forward_mode, hidden_states.shape[0]
        )
        hidden_states = self._forward_with_selection_output(
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
            comm_manager=comm_manager,
            block_scale=block_scale,
            selection=selection,
        )
        if self.computes_selection:
            selection = self._forward_selection
        if selection is None:
            raise RuntimeError("DSA explicit selection was not produced")
        return (hidden_states, selection)

    @break_point
    def _forward_with_selection_output(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        comm_manager: CommManager,
        block_scale: torch.Tensor | None,
        selection: DSASelection | None,
    ) -> torch.Tensor:
        return self._forward_impl(
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
            comm_manager=comm_manager,
            block_scale=block_scale,
            selection=selection,
        )

    def _forward_impl(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        comm_manager: CommManager,
        block_scale: torch.Tensor | None,
        selection: DSASelection | None,
    ) -> torch.Tensor:
        """GLM-5 DSA attention, one COARSE breakable-graph break point.

        Like DeepSeek-V4 it does paged-cache writes, a data-dependent indexer
        -> top-k stage and the FlashMLA sparse kernel (plus pre-attn
        collectives), none capturable. Under a prefill-graph capture the whole
        attention runs eager (reading the live ``ctx``) while the layer's
        norms + MoE stay graphed; direct call otherwise (see ``break_point``).
        Padded token-shaped inputs are sliced to the real count the live
        metadata describes -- DSA and the decode-window split (which derives
        ``decode_start`` from the total token count) must not see padded rows,
        or decode rows get sliced out of the padded tail. Mirrors the
        DeepSeek-V4 DSA break.
        """
        if self.computes_selection:
            if selection is not None:
                raise RuntimeError("DSA owner must compute a fresh selection")
            self._forward_selection.prefill = None
            self._forward_selection.decode = None
        else:
            if selection is None:
                raise RuntimeError("DSA reuse layer requires its owner's selection")
            selection.require_consumer(self.attn_mqa.layer_id)
        if hidden_states.shape[0] == 0:
            return hidden_states
        decode_metadata = getattr(ctx.attn_backend, "forward_decode_metadata", None)
        qkv = self.fused_qkv_a_proj_with_mqa(hidden_states, block_scale, torch.bfloat16)
        _qkv_width = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
        if qkv.shape[-1] != _qkv_width:
            qkv = qkv[..., :_qkv_width]
        qkv = comm_manager.pre_attn_comm(qkv, ctx)
        _metadata = getattr(ctx.attn_backend, "forward_metadata", None)
        _token_to_req = getattr(_metadata, "token_to_req_indices", None)
        if current_forward_ctx() is not None and _token_to_req is not None:
            positions, qkv, out_cache_loc = slice_to_real_tokens(
                _token_to_req.numel(), positions, qkv, out_cache_loc
            )
        q_a, latent_cache = qkv.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        kv_a = latent_cache[..., : self.kv_lora_rank]
        q_norm = torch.empty_like(q_a)
        if q_a.size(0) > 0:
            self.fused_qk_layernorm(input_q_a=q_a, input_kv_a=kv_a, output_q_a=q_norm)
        if self.q_lora_scale is not None:
            q_norm = q_norm * self.q_lora_scale
        if self.kv_lora_scale is not None:
            kv_a.mul_(self.kv_lora_scale)
        num_attn_tokens = int(q_norm.shape[0])
        decode_window = self._resolve_decode_window(
            ctx, decode_metadata, total_tokens=num_attn_tokens
        )
        num_decode_tokens = decode_window.num_tokens
        num_prefill_tokens = decode_window.start
        decode_start = decode_window.start
        decode_end = decode_window.end
        prefill_selection = None if self.computes_selection else selection.prefill
        decode_selection = None if self.computes_selection else selection.decode
        should_compute_indexer = self.computes_selection
        if should_compute_indexer:
            prefill_selection, decode_selection = self._compute_indexer_selections(
                hidden_states=hidden_states,
                q_norm=q_norm,
                positions=positions,
                ctx=ctx,
                out_cache_loc=out_cache_loc,
                comm_manager=comm_manager,
                num_attn_tokens=num_attn_tokens,
                num_prefill_tokens=num_prefill_tokens,
                prefill_selection=prefill_selection,
                decode_selection=decode_selection,
            )
        q = self.q_b_proj(q_norm)[0]
        attn_output = torch.empty(
            q.size(0),
            self.num_local_heads * self.v_head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        if ctx.num_extends > 0:
            prefill_ctx = replace(
                ctx,
                bs=ctx.num_extends,
                input_num_tokens=num_prefill_tokens,
                forward_mode=ForwardMode.EXTEND,
            )
            if prefill_selection is None:
                raise RuntimeError(
                    "GLM DSA sparse prefill requires computed top-k indices."
                )
            self.forward_dsa_sparse_prefill(
                positions[:num_prefill_tokens],
                q[:num_prefill_tokens],
                latent_cache[:num_prefill_tokens],
                prefill_ctx,
                out_cache_loc[:num_prefill_tokens],
                attn_output[:num_prefill_tokens],
                prefill_topk=prefill_selection,
            )
        if num_decode_tokens > 0:
            decode_ctx = replace(
                ctx,
                bs=decode_window.num_reqs,
                num_extends=0,
                input_num_tokens=num_decode_tokens,
                forward_mode=ForwardMode.DECODE,
            )
            if decode_selection is None:
                raise RuntimeError(
                    "GLM DSA sparse decode requires computed top-k indices."
                )
            topk_indices, topk_lens = self._slice_decode_topk(
                decode_selection, decode_start, decode_end
            )
            self.forward_absorb(
                positions[decode_start:decode_end],
                q[decode_start:decode_end],
                latent_cache[decode_start:decode_end],
                decode_ctx,
                out_cache_loc[decode_start:decode_end],
                attn_output[decode_start:decode_end],
                topk_indices=topk_indices,
                topk_lens=topk_lens,
            )
        if ctx.draft_narrowing is not None:
            attn_output = attn_output.index_select(0, ctx.gather_ids)
        output, _ = self.o_proj(attn_output)
        return output

    def _compute_decode_topk_indices_portable(
        self,
        *,
        indexer_output: GlmDsaIndexerOutput,
        ctx: ForwardContext,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        q_len_per_req: int,
        decode_start: int,
        num_tokens: int,
        num_decode_tokens: int,
        topk: int,
    ) -> GlmDsaDecodeTopK:
        q = indexer_output.query[decode_start : decode_start + num_decode_tokens]
        weights = indexer_output.weights[
            decode_start : decode_start + num_decode_tokens
        ]
        index_k_cache = ctx.token_to_kv_pool.get_index_k_buffer(self.attn_mqa.layer_id)
        if index_k_cache is None:
            raise RuntimeError("GLM DSA top-k requires an index-K cache.")
        writes_full_workspace = decode_start == 0 and num_decode_tokens == num_tokens
        topk_indices = self._get_decode_topk_workspace(
            num_tokens, topk, q.device, fill_value=None if writes_full_workspace else -1
        )
        topk_slice = topk_indices[decode_start : decode_start + num_decode_tokens]
        topk_lens = self._get_decode_topk_lens_workspace(
            num_tokens, q.device, fill=not writes_full_workspace
        )
        topk_lens_slice = topk_lens[decode_start : decode_start + num_decode_tokens]
        metadata = ctx.attn_backend.forward_decode_metadata
        seq_lens_2d = (
            metadata._dsa_seq_lens_2d[ctx.num_extends * q_len_per_req :]
            if q_len_per_req > 1
            else seq_lens.unsqueeze(1)
        )
        placement = ctx.attn_backend.cache_placement(self.attn_mqa)
        if placement is not None:
            page_size = ctx.token_to_kv_pool.arena.kv_page_size
            requests = (
                torch.arange(q.shape[0], device=q.device, dtype=torch.int32)
                // q_len_per_req
            )
            causal_lens = (
                seq_lens[requests.long()]
                - (q_len_per_req - 1)
                + torch.arange(q.shape[0], device=q.device) % q_len_per_req
            )
            offsets, counts = select_dsa_topk(
                q,
                weights,
                index_k_cache,
                page_table,
                requests,
                causal_lens,
                placement=placement,
                page_size=page_size,
                topk=topk,
                softmax_scale=(
                    1.0
                    if q.dtype == torch.float8_e4m3fn
                    else self.indexer.weights_softmax_scale
                ),
                max_logits_bytes=64 * 1024 * 1024,
                initial_tokens=self.initial_tokens,
                local_tokens=self.local_tokens,
            )
            pages = page_table[
                requests.long().unsqueeze(1), offsets.clamp_min(0).long() // page_size
            ]
            topk_slice.copy_(
                torch.where(offsets >= 0, pages * page_size + offsets % page_size, -1)
            )
            topk_lens_slice.copy_(counts)
        else:
            dsa_decode_topk(
                q,
                weights,
                seq_lens,
                page_table,
                page_size=ctx.token_to_kv_pool.arena.kv_page_size,
                topk=topk,
                softmax_scale=(
                    1.0
                    if q.dtype == torch.float8_e4m3fn
                    else self.indexer.weights_softmax_scale
                ),
                batch_invariant=False,
                q_len_per_req=q_len_per_req,
                index_k_cache=index_k_cache,
                seq_lens_2d=seq_lens_2d,
                plan=metadata._dsa_plan,
                candidate_policy=longcat_dsa_candidates,
                out=topk_slice,
                lens_out=topk_lens_slice,
                slot_order=global_server_args_dict["dsa_slot_order"],
                initial_tokens=self.initial_tokens,
                local_tokens=self.local_tokens,
            )
        return GlmDsaDecodeTopK(topk_indices=topk_indices, topk_lens=topk_lens)

    def _compute_prefill_topk_indices_portable(
        self,
        *,
        indexer_output: GlmDsaIndexerOutput,
        ctx: ForwardContext,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        kv_workspace_slots: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        candidate_lens_cpu: torch.Tensor,
        kv_seq_lens: torch.Tensor,
        max_seq_len: int,
        num_prefill_tokens: int,
        topk: int,
    ) -> GlmDsaPrefillTopK:
        q = indexer_output.query[:num_prefill_tokens].contiguous()
        weights = indexer_output.weights[:num_prefill_tokens]
        index_k_cache = ctx.token_to_kv_pool.get_index_k_buffer(self.attn_mqa.layer_id)
        if index_k_cache is None:
            raise RuntimeError("GLM DSA top-k requires an index-K cache.")
        max_logits_mb = int(global_server_args_dict[_INDEXER_PREFILL_MAX_LOGITS_MB_ARG])
        placement = ctx.attn_backend.cache_placement(self.attn_mqa)
        if placement is not None:
            global_starts, global_ends = (row_starts, row_ends)
            offsets, topk_lens = select_dsa_topk(
                q,
                weights,
                index_k_cache,
                page_table,
                torch.searchsorted(
                    torch.cumsum(seq_lens.to(device=q.device, dtype=torch.int64), 0),
                    row_starts.long(),
                    right=True,
                ).to(torch.int32),
                global_ends - global_starts,
                placement=placement,
                page_size=ctx.token_to_kv_pool.arena.kv_page_size,
                topk=topk,
                softmax_scale=(
                    1.0
                    if q.dtype == torch.float8_e4m3fn
                    else self.indexer.weights_softmax_scale
                ),
                max_logits_bytes=max(1, max_logits_mb) * 1024 * 1024,
                initial_tokens=self.initial_tokens,
                local_tokens=self.local_tokens,
            )
            workspace_indices = torch.where(
                offsets >= 0, offsets + global_starts.unsqueeze(1), -1
            ).to(torch.int32)
        else:
            workspace_indices, topk_lens = dsa_prefill_topk(
                q,
                weights,
                kv_workspace_slots,
                row_starts.to(torch.int32).contiguous(),
                row_ends.to(torch.int32).contiguous(),
                topk=topk,
                softmax_scale=(
                    1.0
                    if q.dtype == torch.float8_e4m3fn
                    else self.indexer.weights_softmax_scale
                ),
                batch_invariant=False,
                index_k_cache=index_k_cache,
                page_size=ctx.token_to_kv_pool.arena.kv_page_size,
                max_logits_bytes=max(1, max_logits_mb) * 1024 * 1024,
                candidate_lens_cpu=candidate_lens_cpu,
                slot_order=global_server_args_dict["dsa_slot_order"],
                initial_tokens=self.initial_tokens,
                local_tokens=self.local_tokens,
            )
        return GlmDsaPrefillTopK(
            workspace_indices=workspace_indices,
            topk_lens=topk_lens,
            page_table=page_table,
            seq_lens=seq_lens.to(device=q.device, dtype=torch.int32),
            kv_seq_lens=kv_seq_lens,
            max_seq_len=max_seq_len,
            kv_workspace_slots=kv_workspace_slots,
            row_starts=row_starts,
            row_ends=row_ends,
            candidate_lens_cpu=candidate_lens_cpu,
        )

    def _check_decode_q_len_per_req(self, q_len_per_req: int) -> None:
        if not 1 <= q_len_per_req <= self.decode_query_length_limit:
            raise NotImplementedError(
                f"LongCat decode query length {q_len_per_req} exceeds {self.decode_query_length_limit}"
            )

    def forward_absorb_attn_v_proj(
        self, Q, K, ctx, out_cache_loc, output, topk_indices=None, topk_lens=None
    ):
        return super().forward_absorb_attn_v_proj(
            Q, ctx, output, topk_indices=topk_indices, topk_lens=topk_lens
        )

    def forward_absorb(
        self,
        positions,
        q,
        latent_cache,
        ctx,
        out_cache_loc,
        output,
        topk_indices=None,
        topk_lens=None,
    ):
        Q = self.forward_absorb_qkv_proj(q, latent_cache, positions, ctx, out_cache_loc)
        return self.forward_absorb_attn_v_proj(
            Q,
            None,
            ctx,
            out_cache_loc,
            output,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
        )

    def forward(self, positions, hidden_states, ctx, comm_manager, block_scale=None):
        raise RuntimeError(
            "LongCat owner/reuse attention requires forward_with_selection"
        )
