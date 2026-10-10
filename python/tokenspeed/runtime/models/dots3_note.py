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

"""Dots3-note text backbone: Full DSA and sliding-window MLA.

The shared decoder, MoE, model forward and non-attention loader are DeepSeek-V3
implementations. Attention deliberately does not use its fused QKV-A loader.
Small KV-A/gate/index-weight projections are dequantized at load time; the
remaining linears retain the runtime's block-FP8 implementation.

Checkpoint index metadata selects the indexer RoPE weight layout: ``leading``
weights are converted to the runtime's single ``tail`` layout during loading.
Absent layout metadata (including an absent index) means historical tail weights;
unknown layouts fail. The conversion preserves FP8 bytes and 128x128 scale grids.
The target is a multimodal architecture, currently requiring language-model-only
because its encoders are not implemented. Text loading skips only the known
vision/audio encoder prefixes and MTP weights.

Runtime contract (provided by the cache/backend registration):
* ``router.leaf_for(attn_mqa)`` exposes MLA-shaped prefill/decode metadata;
  ``chunked_prefill_metadata.page_table`` uses absolute logical page columns.
* ``router.write_locations(attn_mqa, mode)`` supplies group-local slots, with
  padding directed to the null page. Sparse prefill is routed by ``layer``.
* The pool exposes BF16 ``[pages, page_size, 1, rank + rope]`` latent views to
  the unified MLA prologue, the sole attention KV writer. Index keys have
  their own planar uint8 pages and already quantized values/scales.

SWA prefill gathers only each query tile's visible prefix plus current rows,
then expands KV and uses bottom-right causal window attention. No private
request history or prefix-only LSE merge is maintained. Decode uses the same
refreshed leaf metadata in eager and graphs, including packed target verify rows.
The native NextN model reuses the SWA/dense layer with accepted-row narrowing.
CP, PP and multimodal encoders are not supported by this adapter.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.dsa import dsa_decode_topk, dsa_prefill_topk
from tokenspeed_kernel.ops.attention.dsa.triton import workspace_topk_to_global_slots
from tokenspeed_kernel.ops.attention.mla import mla_prefill
from tokenspeed_kernel.ops.transform import hadamard_transform
from torch import nn

from tokenspeed.runtime.configs.dots3_note import Dots3NoteConfig
from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.execution.breakable_cuda_graph import break_point
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.page_table import (
    build_prefill_kv_workspace_slots,
)
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.logits_processor import should_apply_lm_head_quant_method
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.layers.quantization.utils import block_dequant
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.deepseek_nextn import DeepseekV3DraftDecoderLayer
from tokenspeed.runtime.models.deepseek_v3 import (
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3Model,
    DeepseekV3MoE,
)
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.env import global_server_args_dict

_SWA_QUERY_TILE = 256


def quantize_index_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode BF16 Hadamard rows as E4M3FN and FP32 power-of-two scales.

    Args:
        x: Materialized BF16 rows, with the 128 index channels last. Q has
            one row per token/head; K has one row per token.
    Returns:
        FP8 values of the same shape and FP32 scales with the last axis removed.
        This is not the unrounded ``index_v4`` cache codec.
    """
    if x.dtype != torch.bfloat16 or x.shape[-1] != 128:
        raise ValueError("Index quantization requires BF16 rows of width 128")
    values = x.float()
    amax = values.abs().amax(dim=-1).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (values / scale.unsqueeze(-1)).to(torch.float8_e4m3fn), scale


class Dots3NoteIndexer(nn.Module):
    def __init__(
        self,
        config: Dots3NoteConfig,
        *,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.checkpoint_rope_layout: str = "tail"
        self.wq_b = ReplicatedLinear(
            config.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.wk = ReplicatedLinear(
            config.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wk", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            config.hidden_size,
            self.n_heads,
            bias=False,
            quant_config=None,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6, dtype=torch.bfloat16)
        self.rotary_emb = get_rope(
            self.rope_dim,
            rotary_dim=self.rope_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=None,
            is_neox_style=False,
        )

    def forward(self, x, q_lora, positions):
        """Return BF16 FP8-query values, FP8 keys, K scales and folded weights."""
        q = self.wq_b(q_lora)[0].view(-1, self.n_heads, self.head_dim)
        k = self.wk(x)[0]
        k = F.layer_norm(
            k.float(),
            (self.head_dim,),
            self.k_norm.weight.float(),
            self.k_norm.bias.float(),
            self.k_norm.eps,
        ).to(x.dtype)
        q_rope, k_rope = self.rotary_emb(
            positions, q[..., -self.rope_dim :], k[:, None, -self.rope_dim :]
        )
        q[..., -self.rope_dim :] = q_rope
        k[..., -self.rope_dim :] = k_rope.squeeze(1)
        q = hadamard_transform(q.contiguous(), scale=self.head_dim**-0.5).to(
            torch.bfloat16
        )
        k = hadamard_transform(k.contiguous(), scale=self.head_dim**-0.5).to(
            torch.bfloat16
        )
        q8, q_scale = quantize_index_rows(q)
        k8, k_scale = quantize_index_rows(k)
        # Preserve the signed raw head weights and the reference's FP32 order.
        weights = self.weights_proj(x)[0].float() * self.n_heads**-0.5
        weights = weights * q_scale * self.head_dim**-0.5
        return q8.to(torch.bfloat16), k8, k_scale, weights


class Dots3NoteAttention(nn.Module):
    def __init__(
        self,
        config: Dots3NoteConfig,
        layer_id: int,
        mapping: Mapping,
        *,
        is_nextn: bool,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        if quant_config is not None and (
            not isinstance(quant_config, Fp8Config)
            or quant_config.weight_block_size != [128, 128]
            or not quant_config.is_checkpoint_fp8_serialized
            or quant_config.scale_fmt is not None
        ):
            raise ValueError(
                "dots3_note supports serialized FP8 with FP32 128x128 scales"
            )
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.use_swa = config.layer_types[layer_id] == "sliding_attention"
        self.is_nextn = is_nextn
        if is_nextn and (layer_id != 0 or not self.use_swa):
            raise ValueError("dots3_note NextN requires one SWA layer at cache layer 0")
        if self.use_swa:
            self.num_heads = config.swa_num_attention_heads
            self.q_lora_rank = config.swa_q_lora_rank
            self.kv_lora_rank = config.swa_kv_lora_rank
            self.qk_nope_head_dim = config.swa_qk_nope_head_dim
            self.qk_rope_head_dim = config.swa_qk_rope_head_dim
            self.v_head_dim = config.swa_v_head_dim
            rope_theta = config.swa_rope_theta
        else:
            self.num_heads = config.num_attention_heads
            self.q_lora_rank = config.q_lora_rank
            self.kv_lora_rank = config.kv_lora_rank
            self.qk_nope_head_dim = config.qk_nope_head_dim
            self.qk_rope_head_dim = config.qk_rope_head_dim
            self.v_head_dim = config.v_head_dim
            rope_theta = config.rope_theta
        if self.num_heads % mapping.attn.tp_size:
            raise ValueError("Attention heads must divide attention TP size")
        self.num_local_heads = self.num_heads // mapping.attn.tp_size
        self.head_start = mapping.attn.tp_rank * self.num_local_heads
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.qk_head_dim**-0.5
        self.window_left = config.sliding_window_size - 1 if self.use_swa else -1
        self.index_topk = config.index_topk
        self.q_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("q_a_proj", prefix),
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=None,
            prefix=add_prefix("kv_a_proj_with_mqa", prefix),
        )
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("q_b_proj", prefix),
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("kv_b_proj", prefix),
        )
        self.g_proj = ReplicatedLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=None,
            prefix=add_prefix("g_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.k_rope_only_layernorm = RMSNorm(
            self.qk_rope_head_dim, eps=config.rms_norm_eps
        )
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            rope_scaling=None,
            is_neox_style=False,
        )
        self.attn_mqa = PagedAttention(
            num_heads=self.num_local_heads,
            head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            scaling=self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            sliding_window_size=self.window_left,
            rotary_emb=self.rotary_emb,
            qk_norm=None,  # Latent and key-RoPE normalization precede the prologue.
        )
        self.indexer = (
            None
            if self.use_swa
            else Dots3NoteIndexer(
                config, quant_config=quant_config, prefix=add_prefix("indexer", prefix)
            )
        )
        self.register_buffer("w_kc", None, persistent=False)
        self.register_buffer("w_vc", None, persistent=False)

    def prepare_weights(self) -> None:
        weight = self.kv_b_proj.weight
        if weight.dtype == torch.float8_e4m3fn:
            # SWA's 320 rows/head cross 128-row scale blocks. Dequantize on
            # the original matrix grid BEFORE introducing the head dimension.
            weight = block_dequant(weight, self.kv_b_proj.weight_scale_inv, [128, 128])
        weight = weight.to(torch.bfloat16).view(
            self.num_local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        wk, wv = weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        self.w_kc = wk.contiguous()
        self.w_vc = wv.transpose(1, 2).contiguous()

    def project(self, x):
        """Project and normalize latent rows; the prologue owns RoPE and KV writes."""
        q_lora = self.q_a_layernorm(self.q_a_proj(x)[0])
        q_lora = (q_lora * math.sqrt(self.hidden_size / self.q_lora_rank)).to(x.dtype)
        latent, k_rope = self.kv_a_proj_with_mqa(x)[0].split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        latent = self.kv_a_layernorm(latent)
        latent = (latent * math.sqrt(self.hidden_size / self.kv_lora_rank)).to(x.dtype)
        k_rope = self.k_rope_only_layernorm(k_rope)
        q = self.q_b_proj(q_lora)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        return q_lora, q, latent, k_rope

    def absorb_query(self, q):
        if self.w_kc is None:
            raise RuntimeError("post_load_weights must prepare absorbed MLA weights")
        latent_q = torch.bmm(
            q[..., : self.qk_nope_head_dim].transpose(0, 1), self.w_kc
        ).transpose(0, 1)
        return torch.cat((latent_q, q[..., self.qk_nope_head_dim :]), dim=-1)

    def expand_values(self, values):
        if self.w_vc is None:
            raise RuntimeError("post_load_weights must prepare absorbed MLA weights")
        values = values.view(-1, self.num_local_heads, self.kv_lora_rank)
        return torch.bmm(values.transpose(0, 1), self.w_vc).transpose(0, 1)

    def _swa_prefill(self, q, ctx, leaf):
        meta = leaf.chunked_prefill_metadata
        cache = ctx.token_to_kv_pool.get_key_buffer(self.layer_id)
        page_size = cache.shape[1]
        output = q.new_empty((q.shape[0], self.num_local_heads, self.v_head_dim))
        cu_seqlens = torch.arange(2, dtype=torch.int32, device=q.device)
        query_start = 0
        # ponytail: bounded per-request tiles; batch these only if launch overhead matters.
        for req, (prefix, length) in enumerate(
            zip(
                meta.extend_prefix_lens_cpu[: ctx.num_extends].tolist(),
                meta.extend_seq_lens_cpu[: ctx.num_extends].tolist(),
                strict=True,
            )
        ):
            for start in range(0, length, _SWA_QUERY_TILE):
                end = min(start + _SWA_QUERY_TILE, length)
                first = max(0, prefix + start - self.window_left)
                last = prefix + end
                logical = torch.arange(first, last, device=q.device, dtype=torch.int64)
                pages = meta.page_table[req, logical // page_size].long()
                rows = cache[pages, logical % page_size, 0]
                kv = self.kv_b_proj(rows[:, : self.kv_lora_rank].contiguous())[0]
                kv = kv.view(
                    -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
                )
                k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
                rope = rows[:, None, self.kv_lora_rank :].expand(
                    -1, self.num_local_heads, -1
                )
                k = torch.cat((k_nope, rope), dim=-1)
                output[query_start + start : query_start + end] = mla_prefill(
                    q[query_start + start : query_start + end].contiguous(),
                    k,
                    v.contiguous(),
                    cu_seqlens * (end - start),
                    cu_seqlens * (last - first),
                    end - start,
                    last - first,
                    self.scaling,
                    is_causal=True,
                    window_left=self.window_left,
                    solution=leaf.kernel_solution,
                )
            query_start += length
        if query_start != q.shape[0]:
            raise ValueError("SWA prefill metadata does not cover the query rows")
        if leaf.step_counter is not None:
            leaf.step_counter.record_cache()
        return output

    def _sparse_prefill(self, q, index_q, weights, ctx, leaf):
        meta = leaf.chunked_prefill_metadata
        lengths = meta.extend_seq_lens_cpu[: ctx.num_extends].to(torch.int64)
        prefixes = meta.extend_prefix_lens_cpu[: ctx.num_extends].to(torch.int64)
        totals = lengths + prefixes
        max_len = int(totals.max().item())
        page_size = ctx.token_to_kv_pool.get_key_buffer(self.layer_id).shape[1]
        table = meta.page_table[
            : ctx.num_extends, : (max_len + page_size - 1) // page_size
        ]
        table = table.to(torch.int32).contiguous()
        seq_lens = meta.seq_lens[: ctx.num_extends]
        slots = build_prefill_kv_workspace_slots(
            page_table=table,
            seq_lens=seq_lens,
            max_seq_len=max_len,
            page_size=page_size,
            device=q.device,
            num_tokens=int(totals.sum().item()),
        )
        starts, ends, candidates = [], [], []
        offset = 0
        for prefix, length in zip(prefixes.tolist(), lengths.tolist(), strict=True):
            visible = torch.arange(
                prefix + 1, prefix + length + 1, dtype=torch.int32, device=q.device
            )
            starts.append(
                torch.full((length,), offset, dtype=torch.int32, device=q.device)
            )
            ends.append(visible + offset)
            candidates.append(visible)
            offset += prefix + length
        candidates = torch.cat(candidates)
        indices, topk_lens = dsa_prefill_topk(
            index_q.contiguous(),
            weights,
            slots,
            torch.cat(starts),
            torch.cat(ends),
            topk=self.index_topk,
            softmax_scale=1.0,
            index_k_cache=ctx.token_to_kv_pool.get_index_k_buffer(self.layer_id),
            page_size=page_size,
            max_logits_bytes=64 * 1024 * 1024,
            batch_invariant=leaf.batch_invariant,
            slot_order=leaf.slot_order,
            solution=leaf.kernel_solution,
        )
        result = ctx.attn_backend.forward_sparse_prefill(
            q=q,
            layer=self.attn_mqa,
            token_to_kv_pool=ctx.token_to_kv_pool,
            kv_seq_lens=candidates,
            topk_slots=workspace_topk_to_global_slots(
                workspace_indices=indices, kv_workspace_slots=slots
            ),
            topk_lens=topk_lens,
            max_seq_len=max_len,
        )
        return self.expand_values(result)

    @break_point
    def forward(
        self,
        positions,
        hidden_states,
        ctx: ForwardContext,
        comm_manager: CommManager,
        block_scale: torch.Tensor | None = None,
    ):
        if block_scale is not None:
            raise NotImplementedError(
                "dots3_note attention requires materialized BF16 input"
            )
        narrowing = ctx.draft_narrowing is not None
        if narrowing and not self.is_nextn:
            raise NotImplementedError("Only dots3_note NextN can narrow draft rows")
        if narrowing and (ctx.gather_ids is None or ctx.gather_ids.numel() != ctx.bs):
            raise ValueError("dots3_note NextN requires one gather row per request")
        x = comm_manager.pre_attn_comm(hidden_states, ctx)
        if x.shape[0] == 0:
            return x
        leaf = ctx.attn_backend.leaf_for(self.attn_mqa)
        num_decode = ctx.bs - ctx.num_extends
        num_prefill = 0
        if ctx.num_extends:
            meta = leaf.chunked_prefill_metadata
            if meta is None or meta.page_table is None:
                raise RuntimeError(
                    "dots3_note prefill requires group-local page tables"
                )
            num_prefill = int(meta.extend_seq_lens_cpu[: ctx.num_extends].sum().item())
        q_len_per_req = leaf.forward_decode_metadata.q_len_per_req if num_decode else 1
        # Draft metadata describes one live query, but step 0 still writes the
        # complete verify window published by the router, including rejected rows.
        num_decode_tokens = (
            ctx.attn_backend.write_locations(self.attn_mqa, ForwardMode.DECODE).numel()
            if narrowing and num_decode
            else num_decode * q_len_per_req
        )
        num_tokens = num_prefill + num_decode_tokens
        if num_tokens != ctx.input_num_tokens or num_tokens > x.shape[0]:
            raise ValueError(
                "dots3_note token counts do not match prefill and decode metadata"
            )
        if num_tokens == 0:
            return torch.zeros_like(x)
        if x.dtype != torch.bfloat16:
            raise ValueError("dots3_note attention requires BF16 activations")
        # Collective padding is not a request: only ctx's logical rows write KV.
        x_active = x[:num_tokens]
        positions = positions[:num_tokens]
        q_lora, q, latent, k_rope = self.project(x_active)
        absorbed_q = self.absorb_query(q)
        latent_cache = torch.cat((latent, k_rope), dim=-1)
        index_q = weights = None
        if self.indexer is not None:
            index_q, index_k, index_scale, weights = self.indexer(
                x_active, q_lora, positions
            )
        for start, end, mode in (
            (0, num_prefill, ForwardMode.EXTEND),
            (num_prefill, num_tokens, ForwardMode.DECODE),
        ):
            if start == end:
                continue
            loc = ctx.attn_backend.write_locations(self.attn_mqa, mode)
            if loc.numel() != end - start:
                raise ValueError(
                    "Group write locations do not match dots3_note query rows"
                )
            prepared = self.attn_mqa.latent_prologue(
                absorbed_q[start:end],
                absorbed_q[start:end, :, self.kv_lora_rank :],
                latent_cache[start:end],
                positions[start:end],
                ctx,
                slots=loc,
                expanded=None,
                key_rows=None,
            )
            absorbed_q[start:end].copy_(prepared.query)
            if self.indexer is not None:
                ctx.token_to_kv_pool.set_index_k_buffer(
                    self.layer_id, loc, index_k[start:end], index_scale[start:end]
                )
        if narrowing:
            # KV is complete before the drafter publishes the accepted frontier.
            # Query, gate input and the decoder residual select the same live rows.
            ctx.draft_narrowing.publish_accepted_prefix()
            absorbed_q = absorbed_q.index_select(0, ctx.gather_ids)
            x_active = x_active.index_select(0, ctx.gather_ids)
            decode_ctx = replace(
                ctx,
                num_extends=0,
                input_num_tokens=ctx.bs,
                forward_mode=ForwardMode.DECODE,
            )
            with ctx.attn_backend.override_num_extends(0):
                result = self.attn_mqa(
                    absorbed_q,
                    k=None,
                    v=None,
                    positions=None,
                    ctx=decode_ctx,
                    record_kv_cache=not ctx.forward_mode.is_decode_or_idle(),
                )
            values = self.expand_values(result)
            gate = self.g_proj(x_active)[0][
                :, self.head_start : self.head_start + self.num_local_heads
            ]
            values = values * torch.sigmoid(gate).unsqueeze(-1)
            return self.o_proj(values.flatten(1))[0]

        values = q.new_empty((num_tokens, self.num_local_heads, self.v_head_dim))
        if num_prefill:
            if self.use_swa:
                # Cached SWA expands keys tile by tile. Reuse the prologue's
                # rotated query tail without rotating or storing any row twice.
                prefill_q = torch.cat(
                    (
                        q[:num_prefill, :, : self.qk_nope_head_dim],
                        absorbed_q[:num_prefill, :, self.kv_lora_rank :],
                    ),
                    dim=-1,
                )
                values[:num_prefill] = self._swa_prefill(prefill_q, ctx, leaf)
            else:
                values[:num_prefill] = self._sparse_prefill(
                    absorbed_q[:num_prefill],
                    index_q[:num_prefill],
                    weights[:num_prefill],
                    ctx,
                    leaf,
                )
        if num_decode:
            kwargs = {}
            if self.indexer is not None:
                meta = leaf.forward_decode_metadata
                begin = meta.num_extends
                seq_lens = meta.seq_lens_k[begin : begin + num_decode].contiguous()
                table = meta.page_table[begin : begin + num_decode].contiguous()
                indices, lens = dsa_decode_topk(
                    index_q[num_prefill:].contiguous(),
                    weights[num_prefill:],
                    seq_lens,
                    table,
                    page_size=ctx.token_to_kv_pool.get_key_buffer(self.layer_id).shape[
                        1
                    ],
                    topk=self.index_topk,
                    softmax_scale=1.0,
                    q_len_per_req=q_len_per_req,
                    batch_invariant=leaf.batch_invariant,
                    slot_order=leaf.slot_order,
                    solution=leaf.kernel_solution,
                    index_k_cache=ctx.token_to_kv_pool.get_index_k_buffer(
                        self.layer_id
                    ),
                )
                kwargs = {"topk_indices": indices, "topk_lens": lens}
            decode_ctx = replace(
                ctx,
                bs=num_decode,
                num_extends=0,
                input_num_tokens=num_decode_tokens,
                forward_mode=ForwardMode.DECODE,
            )
            result = self.attn_mqa(
                absorbed_q[num_prefill:],
                k=None,
                v=None,
                positions=None,
                ctx=decode_ctx,
                **kwargs,
            )
            values[num_prefill:] = self.expand_values(result)
        gate = self.g_proj(x_active)[0][
            :, self.head_start : self.head_start + self.num_local_heads
        ]
        values = values * torch.sigmoid(gate).unsqueeze(-1)
        output = x.new_zeros((x.shape[0], self.hidden_size))
        output[:num_tokens] = self.o_proj(values.flatten(1))[0]
        return output


class Dots3NoteDecoderLayer(DeepseekV3DraftDecoderLayer):
    # The shared forward narrows residuals only when ctx.draft_narrowing is set;
    # ordinary target forwards keep every row.
    def __init__(
        self, config, layer_id, mapping, *, is_nextn, quant_config, prefix, alt_stream
    ):
        nn.Module.__init__(self)
        self.mapping = mapping
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.is_moe_layer = not is_nextn and self._is_moe_layer(layer_id, False, config)
        dense_batch_invariant = (
            global_server_args_dict["tp_batch_invariant"] == "attn+dense"
        )
        self.self_attn = Dots3NoteAttention(
            config,
            layer_id,
            mapping,
            is_nextn=is_nextn,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        if self.is_moe_layer:
            self.mlp = DeepseekV3MoE(
                config=config,
                mapping=mapping,
                quant_config=quant_config,
                layer_index=layer_id,
                prefix=add_prefix("mlp", prefix),
                alt_stream=alt_stream,
            )
        else:
            self.mlp = DeepseekV3MLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
                mapping=mapping,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                batch_invariant=dense_batch_invariant,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_id,
            is_moe=self.is_moe_layer,
            prev_is_moe=not is_nextn
            and self._is_moe_layer(layer_id - 1, False, config),
            input_layernorm=self.input_layernorm,
            post_attn_layernorm=self.post_attention_layernorm,
            dense_batch_invariant=dense_batch_invariant and not self.is_moe_layer,
            query_sharded=False,
        )


class Dots3NoteModel(DeepseekV3Model):
    def __init__(self, config, mapping, *, quant_config, prefix):
        nn.Module.__init__(self)
        if (
            mapping.attn.qcp_size != 1
            or mapping.attn.dcp_size != 1
            or mapping.pp_size != 1
        ):
            raise NotImplementedError(
                "dots3_note currently supports neither QCP, DCP nor PP"
            )
        self.mapping = mapping
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.alt_stream = torch.cuda.Stream()
        self.layers = nn.ModuleList(
            [
                Dots3NoteDecoderLayer(
                    config,
                    layer_id,
                    mapping,
                    is_nextn=False,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                    alt_stream=self.alt_stream,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture: set[int] = set()
        self._dflash_capture_idx_map: dict[int, int] = {}


class _Dots3NoteForCausalLM(DeepseekV3ForCausalLM):
    """Text loading and startup shared by the multimodal target and text-only MTP."""

    def prepare_communication_runtime(self, max_num_tokens: int) -> bool:
        """Prepare logits after head sharing, before cache planning or serving.

        Returns whether a custom collective is available. The existing logits
        initializers own eligibility and cache both successful and failed probes;
        their capacities are fixed, independent of max_num_tokens.
        """
        processor = self.logits_processor
        if processor.tp_size == 1 or processor.skip_all_gather:
            return False

        if processor._dist_argmax_state is processor._LOGITS_DIST_ARGMAX_UNINITIALIZED:
            processor._dist_argmax_state = (
                processor._init_dist_argmax_state(self.lm_head)
                if processor.do_argmax
                else None
            )
        if processor._all_gather_state is processor._LOGITS_AG_STATE_UNINITIALIZED:
            logits_dtype = (
                self.model.embed_tokens.weight.dtype
                if should_apply_lm_head_quant_method(
                    self.lm_head, self.lm_head.quant_method
                )
                else self.lm_head.weight.dtype
            )
            # Argmax can fall back for softcapping or batches above its capacity.
            processor._all_gather_state = (
                processor._init_all_gather_state(self.lm_head)
                if logits_dtype == torch.bfloat16
                else None
            )
        return (
            processor._dist_argmax_state is not None
            or processor._all_gather_state is not None
        )

    def bind_checkpoint_dir(self, checkpoint_dir: str) -> None:
        """Read indexer layout from a checkpoint directory; missing metadata is tail."""
        index_path = Path(checkpoint_dir) / "model.safetensors.index.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else {}
        metadata = index.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("dots3_note checkpoint index metadata must be an object")
        if (
            "indexer_rope_converted_from" in metadata
            and "indexer_rope_layout" not in metadata
        ):
            raise ValueError("Converted dots3_note weights require indexer_rope_layout")
        layout = metadata.get("indexer_rope_layout", "tail")
        if layout not in ("tail", "leading"):
            raise ValueError(f"Unsupported dots3_note indexer_rope_layout: {layout!r}")
        # NextN has no indexer and need not initialize target-only loader state.
        for layer in self.model.layers:
            indexer = layer.self_attn.indexer
            if indexer is not None:
                indexer.checkpoint_rope_layout = layout

    @staticmethod
    def checkpoint_weight_name_filter(name: str) -> bool:
        """Return whether a checkpoint tensor name belongs in the target text load."""
        return not name.startswith(
            ("vision_encoder.", "audio_encoder.", "model.mtp.", "model.layers.46.")
        )

    def get_param(self, params_dict, name):
        # Unlike the base loader, never turn an unknown backbone weight into a warning.
        return params_dict[name]

    @torch.no_grad()
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        pending: dict[str, dict[str, torch.Tensor]] = {}
        fp8_weights: set[str] = set()
        fp8_scales: set[str] = set()

        def attention_filtered_weights():
            for name, weight in weights:
                # NextN's override filters source names, not its remapped layer 0.
                if not _Dots3NoteForCausalLM.checkpoint_weight_name_filter(name):
                    continue
                if ".self_attn." not in name:
                    mapped_name = name.replace(".gate_proj.", ".gate_up_proj.").replace(
                        ".up_proj.", ".gate_up_proj."
                    )
                    expert_parts = name.split(".")
                    is_expert = (
                        len(expert_parts) == 8
                        and expert_parts[:2] == ["model", "layers"]
                        and expert_parts[2].isdigit()
                        and 0 <= int(expert_parts[2]) < self.config.num_hidden_layers
                        and expert_parts[3:5] == ["mlp", "experts"]
                        and expert_parts[5].isdigit()
                        and 0 <= int(expert_parts[5]) < self.config.n_routed_experts
                        and expert_parts[6] in ("gate_proj", "up_proj", "down_proj")
                        and expert_parts[7] in ("weight", "weight_scale_inv")
                    )
                    if mapped_name not in params and not is_expert:
                        raise KeyError(
                            f"Unsupported dots3_note checkpoint weight: {name}"
                        )
                    yield name, weight
                    continue
                module_name, field = name.rsplit(".", 1)
                module = self.get_submodule(module_name)
                if ".indexer." in module_name:
                    indexer_name, projection = module_name.rsplit(".", 1)
                    indexer = self.get_submodule(indexer_name)
                    if projection in ("wk", "wq_b") and field == "weight_scale_inv":
                        shape = tuple((n + 127) // 128 for n in module.weight.shape)
                        if (
                            weight.dtype != torch.float32
                            or tuple(weight.shape) != shape
                        ):
                            raise ValueError(
                                f"Expected FP32 128x128 indexer scale grid {shape}: {name}"
                            )
                    if indexer.checkpoint_rope_layout == "leading" and (
                        (projection in ("wk", "wq_b") and field == "weight")
                        or (projection == "k_norm" and field in ("weight", "bias"))
                    ):
                        if (
                            indexer.head_dim != 128
                            or indexer.rope_dim != 64
                            or weight.shape != params[name].shape
                        ):
                            raise ValueError(
                                f"Invalid leading indexer weight geometry: {name}"
                            )
                        # Each 128-row head is one FP8 scale block: swapping its
                        # 64-row halves leaves the scale grid unchanged.
                        rows = weight.reshape(-1, 128, *weight.shape[1:])
                        weight = torch.cat(
                            (rows[:, 64:], rows[:, :64]), dim=1
                        ).reshape_as(weight)
                if field == "weight" and weight.dtype == torch.float8_e4m3fn:
                    fp8_weights.add(module_name)
                if field == "weight_scale_inv":
                    fp8_scales.add(module_name)
                small_projection = module_name.endswith(
                    (
                        ".kv_a_proj_with_mqa",
                        ".g_proj",
                        ".indexer.weights_proj",
                    )
                )
                if small_projection and (
                    field == "weight_scale_inv" or weight.dtype == torch.float8_e4m3fn
                ):
                    pair = pending.setdefault(module_name, {})
                    pair[field] = weight
                    if "weight" in pair and "weight_scale_inv" in pair:
                        dequant = block_dequant(
                            pair["weight"], pair["weight_scale_inv"], [128, 128]
                        )
                        module.weight_loader(
                            module.weight, dequant.to(module.weight.dtype)
                        )
                        del pending[module_name]
                    continue
                param = params[name]
                if weight.dtype == torch.float8_e4m3fn and param.dtype != weight.dtype:
                    raise ValueError(f"FP8 quant_config required to load {name}")
                if isinstance(module, LinearBase):
                    module.weight_loader(param, weight)
                else:
                    default_weight_loader(param, weight)
            if pending or fp8_weights != fp8_scales:
                missing = sorted(set(pending) | (fp8_weights ^ fp8_scales))
                raise ValueError(
                    f"Incomplete dots3_note attention FP8 weight/scale pairs: {missing}"
                )

        # The base loader drives this generator, then calls our post-load hook.
        super().load_weights(attention_filtered_weights())

    def post_load_weights(self):
        for layer in self.model.layers:
            layer.self_attn.prepare_weights()


class Dot3NoteForCausalLM(_Dots3NoteForCausalLM):
    """Multimodal target; only explicit language-model-only loading is supported."""

    model_cls = Dots3NoteModel

    def __init__(
        self,
        config: Dots3NoteConfig,
        mapping: Mapping,
        model: Dots3NoteModel | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        is_multimodal_active: bool,
        mm_attention_backend: str | None,
    ) -> None:
        if is_multimodal_active or mm_attention_backend is not None:
            raise ValueError(
                "dots3_note encoders are not implemented; use --language-model-only "
                "without a multimodal attention backend"
            )
        super().__init__(
            config=config,
            mapping=mapping,
            model=model,
            quant_config=quant_config,
            prefix=prefix,
        )


class Dots3NoteForCausalLM(Dot3NoteForCausalLM):
    """Public checkpoint architecture alias."""


EntryClass = [Dot3NoteForCausalLM, Dots3NoteForCausalLM]
