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

"""Ascend DSA model components, fused decode and native projection weight layouts."""

from __future__ import annotations

import math
import weakref
from typing import Any

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.attention.dsa import (
    dsa_fused_prologue,
    dsa_project_output_partial,
    prepare_dsa_projection_weight,
    prepare_dsa_resources,
)
from tokenspeed_kernel.ops.attention.dsa.ascend import (
    supports_fused_dsa_decode,
)
from tokenspeed_kernel.ops.attention.prologue import mla_prologue
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
from tokenspeed.runtime.layers.attention.backends.paged.ascend_dsa import (
    AscendDSABackend,
    ascend_dsa_prefill_backend,
    prepare_ascend_dsa_cp_metadata,
)
from tokenspeed.runtime.layers.attention.mla.weights import (
    _prepare_mla_kv_b_proj_weights,
)
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import get_rope, yarn_get_mscale
from tokenspeed.runtime.utils import add_prefix


def _nk(weight, widths, kl1=512):
    # Bit-preserving private NK panel stream; no dtype conversion.
    assert sum(widths) == weight.shape[0] and weight.shape[1] % kl1 == 0
    panels = []
    col = 0
    for width in widths:
        for start in range(0, weight.shape[1], kl1):
            tile = weight[col : col + width, start : start + kl1]
            panels.append(
                tile.reshape(width // 16, 16, kl1 // 16, 16)
                .permute(2, 0, 1, 3)
                .contiguous()
                .view(-1)
            )
        col += width
    return torch.cat(panels)


def pack_qb_panels(qb):
    """160 uniform N128 tiles; NZ carrier restores the exact raw stream."""
    if qb.device.type != "cpu" or tuple(qb.shape) != (20480, 1536):
        raise ValueError("Expected CPU QB [20480,1536]")
    packed = _nk(qb, [128] * 160)
    return packed.reshape(-1, 16, 16).permute(1, 0, 2).contiguous().reshape(16, 1966080)


def pack_down_qa_panels(down):
    """Sixteen N96 QA tiles plus the original genuine-NZ auxiliary suffix."""
    if down.device.type != "cpu" or tuple(down.shape) != (2240, 4096):
        raise ValueError("Expected CPU down [2240,4096]")
    qa = _nk(down[:1536], [96] * 16)
    aux = down[1536:].reshape(44, 16, 256, 16).permute(0, 2, 3, 1).contiguous().view(-1)
    packed = torch.cat((qa, aux))
    return (
        packed.reshape(140, 256, 16, 16)
        .permute(1, 2, 0, 3)
        .contiguous()
        .reshape(4096, 2240)
    )


def pack_score_fp32(weight):
    """Opaque ND buffer containing FP32 NZ_KN coefficients, bit preserving."""
    if tuple(weight.shape) != (64, 4096) or weight.dtype != torch.float32:
        raise ValueError("FP32 score pack requires [64,4096] float32 weights")
    return (
        weight.reshape(8, 8, 256, 16).permute(0, 2, 3, 1).contiguous().reshape(64, 4096)
    )


class FusedDSADecode:
    def __init__(self, layer, *, rank, group, eps_q, eps_kv, owner=None):
        if layer.num_local_heads != 8 or layer.indexer.index_n_heads != 64:
            raise ValueError(
                "Fused projections require eight local and 64 global heads"
            )
        self.owner = owner
        self.rank = rank
        self.layer_id = layer.layer_id
        self.attention_layer = layer.attn_mqa
        self.initial_tokens = layer.index_init_tokens
        self.window_tokens = layer.index_local_tokens
        self.eps_q, self.eps_kv = eps_q, eps_kv
        self.eps_index = layer.indexer.k_norm.variance_epsilon

        def gathered(value):
            value = value.detach().contiguous()
            outputs = [torch.empty_like(value) for _ in range(8)]
            dist.all_gather(outputs, value, group=group)
            return torch.cat(outputs, 0)

        qb = gathered(layer.q_b_proj.weight).cpu().view(64, 192, 1536)
        if layer.scale_q_lora:
            qb = qb * math.sqrt(layer.hidden_size / layer.q_lora_rank)
        iqb = layer.indexer.wq_b.weight.detach().cpu().view(64, 128, 1536)
        self.qb = prepare_dsa_projection_weight(
            pack_qb_panels(torch.cat((qb, iqb), 1).reshape(20480, 1536)).npu()
        )
        down = torch.cat(
            (
                layer.q_a_proj.weight.detach(),
                layer.kv_a_proj_with_mqa.weight.detach(),
                layer.indexer.wk.weight.detach(),
            ),
            0,
        )
        self.down = prepare_dsa_projection_weight(
            pack_down_qa_panels(down.detach().cpu()).npu()
        )
        self.uk = gathered(layer.w_kc)
        self.gamma_q = layer.q_a_layernorm.weight.detach().contiguous()
        self.gamma_kv = layer.kv_a_layernorm.weight.detach().contiguous()
        self.gamma_index = layer.indexer.k_norm.weight.detach().contiguous()
        self.score_weight = pack_score_fp32(
            (
                layer.indexer.weights_proj.weight.detach().float()
                * layer.indexer.weights_softmax_scale
            ).contiguous()
        )
        self.w_vc = layer.w_vc.detach().contiguous()
        self.w_o = layer.o_proj.weight.detach().contiguous()
        self.rotary_mode = int(layer.rotary_emb is not None) | (
            int(layer.indexer.use_rotary) << 1
        )
        self.rotary_emb = (
            layer.rotary_emb
            if layer.rotary_emb is not None
            else layer.indexer.rotary_emb
        )
        if self.rotary_emb is not None:
            if self.rotary_emb.is_neox_style or self.rotary_emb.rotary_dim != 64:
                raise ValueError("Fused DSA RoPE requires 64 adjacent-pair channels")
            self.cos_sin = self.rotary_emb.cos_sin_cache.float().contiguous().npu()
        else:
            self.cos_sin = None
        self.window = None
        self.outputs = {
            rows: tuple(
                torch.empty(shape, dtype=torch.bfloat16, device="npu")
                for shape in (
                    (8, rows, 8, 576),
                    (rows, 64, 128),
                    (rows, 64),
                )
            )
            for rows in (64, 128, 256, 512)
        }

    def prolog(self, hidden, positions, ctx):
        if self.owner is not None:
            self.owner.bind_backend(ctx.attn_backend)
        stream = torch.npu.current_stream()
        hidden.record_stream(stream)
        if positions is not None:
            positions.record_stream(stream)
        if self.window is None:
            raise RuntimeError("Initialize DSA transport before forward/capture")
        query, index_query, score = self.outputs[hidden.shape[0]]
        pool = ctx.token_to_kv_pool
        placement = ctx.attn_backend.cache_placement(self.attention_layer)
        locations = ctx.attn_backend.write_locations(
            self.attention_layer, ctx.forward_mode
        )
        slots = prepare_ascend_dsa_cp_metadata(
            locations,
            placement,
            initial_tokens=self.initial_tokens,
            window_tokens=self.window_tokens,
        )
        kv_buffer = pool.get_key_buffer(self.layer_id)
        if kv_buffer.dtype != torch.bfloat16:
            raise ValueError("Native DSA requires BF16 KV cache")
        kv = kv_buffer.view(-1, 576)
        index_k = pool.get_component(self.layer_id, "dsa_index_k").view(-1, 128)
        if self.rotary_mode:
            if (
                positions is None
                or positions.ndim != 1
                or positions.numel() != hidden.shape[0]
                or positions.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError(
                    "RoPE positions must be integer indices for all gathered DSA rows"
                )
            positions = positions.to(dtype=torch.int64).contiguous()
        else:
            positions = None
        dsa_fused_prologue(
            hidden.contiguous(),
            self.down,
            self.gamma_q,
            self.gamma_kv,
            self.gamma_index,
            self.score_weight,
            self.qb,
            self.uk,
            self.window,
            kv,
            index_k,
            slots,
            eps_q=self.eps_q,
            eps_kv=self.eps_kv,
            eps_index=self.eps_index,
            positions=positions,
            cos_sin=self.cos_sin,
            rotary_mode=self.rotary_mode,
            out=(query, index_query, score),
        )
        return query, index_query, score

    def output_projection(self, output):
        return dsa_project_output_partial(output, self.w_vc, self.w_o)


class AscendDSAIndexer(nn.Module):
    """DSA indexer with BF16 keys and an FP32 score projection."""

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.index_n_heads = config.index_n_heads
        self.use_rotary = bool(getattr(config, "index_use_rope", False))
        self.rotary_emb = (
            get_rope(
                64,
                rotary_dim=64,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                rope_scaling=config.rope_scaling,
                is_neox_style=False,
            )
            if self.use_rotary
            else None
        )
        self.index_head_dim = config.index_head_dim
        self.weights_softmax_scale = self.index_head_dim**-0.5 * (
            self.index_n_heads**-0.5
        )
        self.wq_b = ReplicatedLinear(
            q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.k_norm = RMSNorm(self.index_head_dim, eps=config.rms_norm_eps)
        self.wk = ReplicatedLinear(
            hidden_size,
            self.index_head_dim,
            bias=False,
            quant_config=None,
            prefix=add_prefix("wk", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.index_n_heads,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=add_prefix("weights_proj", prefix),
        )


class AscendDSAAttention(nn.Module):
    """DSA projections and attention with one independent indexer per layer."""

    rope_is_neox_style = False

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
        layer_id: int | None = None,
        prefix: str = "",
        reduce_attn_results: bool = True,
        *,
        lora_norm_eps: float,
        prefill_enabled: bool,
    ) -> None:
        super().__init__()
        if mapping.attn.has_head_tp:
            raise NotImplementedError(
                "Ascend DSA does not implement the head-TP token exchange"
            )
        if q_lora_rank is None:
            raise ValueError("Ascend DSA requires q_lora_rank")
        self.use_output_gate = bool(getattr(config, "mla_use_output_gate", False))
        if self.use_output_gate:
            raise ValueError("Native DSA requires no output gate")
        self.mapping = mapping
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.scale_q_lora = bool(getattr(config, "mla_scale_q_lora", False))
        self.scale_kv_lora = bool(getattr(config, "mla_scale_kv_lora", False))
        parallel = mapping.attn
        if num_heads % parallel.head_tp_size:
            raise ValueError("DSA heads must divide the attention TP group")
        self.num_local_heads = num_heads // parallel.head_tp_size
        self.q_a_proj = ReplicatedLinear(
            hidden_size,
            q_lora_rank,
            bias=False,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("q_a_proj", prefix),
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            bias=False,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("kv_a_proj_with_mqa", prefix),
        )
        self.q_b_proj = ColumnParallelLinear(
            q_lora_rank,
            num_heads * self.qk_head_dim,
            bias=False,
            params_dtype=torch.bfloat16,
            tp_rank=parallel.head_tp_rank,
            tp_size=parallel.head_tp_size,
            tp_group=parallel.head_tp_group,
            prefix=add_prefix("q_b_proj", prefix),
        )
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            quant_config=quant_config,
            tp_rank=parallel.head_tp_rank,
            tp_size=parallel.head_tp_size,
            tp_group=parallel.head_tp_group,
            prefix=add_prefix("kv_b_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            reduce_results=reduce_attn_results,
            quant_config=quant_config,
            tp_rank=parallel.head_tp_rank,
            tp_size=parallel.head_tp_size,
            tp_group=parallel.head_tp_group,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=lora_norm_eps)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=lora_norm_eps)
        self.scaling = self.qk_head_dim**-0.5
        self.rotary_emb = None
        if not bool(getattr(config, "mla_use_nope", False)):
            if rope_scaling:
                rope_scaling = {**rope_scaling, "rope_type": "deepseek_yarn"}
                mscale = yarn_get_mscale(
                    rope_scaling["factor"],
                    float(rope_scaling.get("mscale_all_dim", False)),
                )
                self.scaling *= mscale * mscale
            self.rotary_emb = get_rope(
                qk_rope_head_dim,
                rotary_dim=qk_rope_head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
                is_neox_style=False,
            )
        self.attn_mqa = PagedAttention(
            self.num_local_heads,
            kv_lora_rank + qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=kv_lora_rank,
            rotary_emb=self.rotary_emb,
            qk_norm=None,
        )
        self.w_kc = self.w_vc = None
        self.index_topk = int(config.index_topk)
        self.index_init_tokens = int(config.index_init_tokens)
        self.index_local_tokens = int(config.index_local_tokens)
        self.indexer = AscendDSAIndexer(
            config=config,
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            quant_config=quant_config,
            prefix=add_prefix("indexer", prefix),
        )
        self._scales_prepared = False
        self._fused_decode_runtime = None
        self._prefill_weights = None
        self.prefill_enabled = prefill_enabled

    def configure_fused_decode(self, runtime) -> None:
        """Attach prepared weights and communication state outside forward."""
        self._fused_decode_runtime = runtime

    def process_weights_after_loading(self, _module=None):
        if not self._scales_prepared:
            # The indexer consumes the unscaled normalized Q latent. Apply
            # Q-LoRA scaling after the main projection, never to this shared norm.
            if self.scale_kv_lora:
                self.kv_a_layernorm.weight.data.mul_(
                    math.sqrt(self.hidden_size / self.kv_lora_rank)
                )
            self._scales_prepared = True
            self.w_kc, self.w_vc = _prepare_mla_kv_b_proj_weights(
                self.kv_b_proj.weight, self
            )

        if self.prefill_enabled and self._prefill_weights is None:
            group = self.mapping.attn.head_tp_group
            pg = (
                process_group_manager.get_device_process_group(group)
                if len(group) > 1
                else None
            )

            def gather(value, dim):
                if len(group) == 1:
                    return value.contiguous()
                pieces = [torch.empty_like(value) for _ in group]
                dist.all_gather(pieces, value.contiguous(), group=pg)
                return torch.cat(pieces, dim=dim).contiguous()

            # Token-local Prefill computes all heads and the complete O row.
            # Decode continues to use the original sharded weights.
            self._prefill_weights = (
                gather(self.q_b_proj.weight, 0),
                gather(self.w_kc, 0),
                gather(self.w_vc, 0),
                gather(self.o_proj.weight, 1),
            )

    def _forward_fused_decode(self, positions, hidden_states, ctx, comm_manager):
        if hidden_states.shape[0] == 0:
            return hidden_states
        if self._fused_decode_runtime is None:
            raise RuntimeError("Initialize native DSA resources before forward/capture")
        if (
            hidden_states.ndim != 2
            or hidden_states.shape[0] not in (64, 128, 256, 512)
            or hidden_states.shape[1] != 4096
            or not ctx.forward_mode.is_decode()
            or ctx.num_extends != 0
            or ctx.bs != hidden_states.shape[0] // 8
        ):
            raise ValueError(
                "Native DSA requires decode-only [64/128/256/512,4096] "
                "and draft8; the requested shape is unsupported"
            )
        packed_query, index_query, index_weights = self._fused_decode_runtime.prolog(
            hidden_states, positions, ctx
        )
        out = ctx.attn_backend.forward(
            packed_query[self._fused_decode_runtime.rank],
            None,
            None,
            self.attn_mqa,
            ctx.token_to_kv_pool,
            ctx.forward_mode,
            ctx.bs,
            save_kv_cache=False,
            record_kv_cache=False,
            fused=True,
            index_query=index_query,
            index_key=None,
            index_weights=index_weights,
            index_init_tokens=self.index_init_tokens,
            index_local_tokens=self.index_local_tokens,
            head_major_output=True,
            index_cache_written=True,
            packed_query=packed_query,
        )
        return self._fused_decode_runtime.output_projection(out)

    def _rotate_indexer(self, positions, index_query, index_key):
        if self.indexer.rotary_emb is not None:
            # The Indexer rotary module describes only the first 64 dimensions,
            # not the full 128-dimensional head. Preserve the NoPE half.
            rotary_dim = self.indexer.rotary_emb.rotary_dim
            query_rope, key_rope = self.indexer.rotary_emb(
                positions,
                index_query[..., :rotary_dim].contiguous().flatten(1),
                index_key[..., :rotary_dim].contiguous().flatten(1),
            )
            index_query = torch.cat(
                (
                    query_rope.view(*index_query.shape[:-1], rotary_dim),
                    index_query[..., rotary_dim:],
                ),
                dim=-1,
            )
            index_key = torch.cat(
                (
                    key_rope.view(*index_key.shape[:-1], rotary_dim),
                    index_key[..., rotary_dim:],
                ),
                dim=-1,
            )
        return index_query, index_key

    def _forward_prefill(self, positions, hidden, ctx, *, backend=None):
        if self._prefill_weights is None:
            raise RuntimeError("Prepare token-local DSA Prefill weights before forward")
        if backend is None:
            backend = ascend_dsa_prefill_backend(ctx.attn_backend)
        plan = backend.prefill_cache_metadata
        if hidden.shape[0] != plan.query_end - plan.query_start:
            raise ValueError(
                "DSA Prefill hidden rows must follow the CP query partition"
            )
        positions = positions[plan.query_start : plan.query_end]
        qb, kc, vc, wo = self._prefill_weights
        indexer = self.indexer
        if hidden.shape[0]:
            q_lora = self.q_a_layernorm(self.q_a_proj(hidden)[0])
            iq = indexer.wq_b(q_lora)[0].view(
                -1, indexer.index_n_heads, indexer.index_head_dim
            )
            ik = indexer.k_norm(indexer.wk(hidden)[0]).unsqueeze(1)
            iq, ik = self._rotate_indexer(positions, iq, ik)
        else:
            iq = hidden.new_empty((0, indexer.index_n_heads, indexer.index_head_dim))
            ik = hidden.new_empty((0, 1, indexer.index_head_dim))
        if hidden.shape[0]:
            # Keep Index-K ready to pack with KV once the latent is formed.
            weights = (
                indexer.weights_proj(hidden.float())[0] * indexer.weights_softmax_scale
            )
            kv = self.kv_a_proj_with_mqa(hidden)[0]
            q = F.linear(q_lora, qb).view(
                -1, kc.shape[0], self.qk_nope_head_dim + self.qk_rope_head_dim
            )
            if self.scale_q_lora:
                q = q * math.sqrt(self.hidden_size / self.q_lora_rank)
            latent = self.kv_a_layernorm(kv[:, : self.kv_lora_rank].contiguous())
            query = torch.cat(
                (
                    torch.bmm(
                        q[..., : self.qk_nope_head_dim].transpose(0, 1), kc
                    ).transpose(0, 1),
                    q[..., self.qk_nope_head_dim :],
                ),
                dim=-1,
            )
            latent = torch.cat((latent, kv[:, self.kv_lora_rank :]), dim=-1)
            prologue = mla_prologue(
                query,
                query[..., self.kv_lora_rank :],
                latent,
                expanded=None,
                rotary=(
                    None
                    if self.rotary_emb is None
                    else self.rotary_emb.as_rotary(positions)
                ),
                cache=None,
                solution=None,
                override=None,
            )
            query, latent = prologue.query, prologue.latent
            if ctx.token_to_kv_pool.latent_write_sanitizes:
                latent = torch.nan_to_num(latent)
        else:
            weights = torch.empty(
                (0, indexer.index_n_heads), dtype=torch.float32, device=hidden.device
            )
            query = hidden.new_empty(
                (0, kc.shape[0], self.kv_lora_rank + self.qk_rope_head_dim)
            )
            latent = hidden.new_empty((0, self.kv_lora_rank + self.qk_rope_head_dim))
        from tokenspeed_kernel.ops.attention.prologue import gathered_latent_store

        backend.prepare_prefill_current(
            self.layer_id,
            latent.contiguous(),
            ik.contiguous(),
            ctx.attn_backend.padded_write_locations(
                self.attn_mqa, ctx.forward_mode, sum(plan.projection_row_counts)
            ),
        )
        indices = backend.select_prefill_indices(
            iq,
            weights,
            self.attn_mqa,
            initial=self.index_init_tokens,
            window=self.index_local_tokens,
        )
        gathered_latent_store(backend.prefill_store(self.attn_mqa))
        output = ctx.attn_backend.forward(
            query,
            None,
            None,
            self.attn_mqa,
            ctx.token_to_kv_pool,
            ctx.forward_mode,
            ctx.bs,
            fused=False,
            save_kv_cache=False,
            record_kv_cache=False,
            prefill_indices=indices,
            head_major_output=True,
        )
        if hidden.shape[0] == 0:
            return hidden
        projected = torch.bmm(output, vc).transpose(0, 1).contiguous()
        return F.linear(projected.flatten(1), wo)

    def forward(
        self, positions, hidden_states, ctx, comm_manager, *, prefill_backend=None
    ):
        if ctx.forward_mode.is_extend_or_mixed():
            if not self.prefill_enabled:
                raise RuntimeError(
                    "DSA Prefill weights were disabled for this decode-only model"
                )
            return self._forward_prefill(
                positions, hidden_states, ctx, backend=prefill_backend
            )
        if hidden_states.shape[0] == 0:
            return hidden_states
        if (
            self._fused_decode_runtime is not None
            and hidden_states.ndim == 2
            and hidden_states.shape[0] in (64, 128, 256, 512)
            and hidden_states.shape[1] == 4096
            and ctx.forward_mode.is_decode()
            and ctx.num_extends == 0
            and ctx.bs == hidden_states.shape[0] // 8
        ):
            return self._forward_fused_decode(
                positions, hidden_states, ctx, comm_manager
            )
        if self.w_kc is None:
            raise RuntimeError("Prepare DSA weights before forward/capture")
        # CommManager has already gathered this attention group's input rows.
        q_lora = self.q_a_layernorm(self.q_a_proj(hidden_states)[0])
        kv = self.kv_a_proj_with_mqa(hidden_states)[0]
        q = self.q_b_proj(q_lora)[0].view(
            -1, self.num_local_heads, self.qk_nope_head_dim + self.qk_rope_head_dim
        )
        if self.scale_q_lora:
            q = q * math.sqrt(self.hidden_size / self.q_lora_rank)
        latent = self.kv_a_layernorm(kv[:, : self.kv_lora_rank].contiguous())
        query = torch.cat(
            (
                torch.bmm(
                    q[..., : self.qk_nope_head_dim].transpose(0, 1), self.w_kc
                ).transpose(0, 1),
                q[..., self.qk_nope_head_dim :],
            ),
            dim=-1,
        )
        latent_cache = torch.cat((latent, kv[:, self.kv_lora_rank :]), dim=-1)
        slots = ctx.attn_backend.padded_write_locations(
            self.attn_mqa, ctx.forward_mode, query.shape[0]
        )
        query = self.attn_mqa.latent_prologue(
            query,
            query[..., self.kv_lora_rank :],
            latent_cache,
            positions,
            ctx,
            slots=slots,
            expanded=None,
            key_rows=None,
        ).query
        indexer = self.indexer
        index_query = indexer.wq_b(q_lora)[0].view(
            -1, indexer.index_n_heads, indexer.index_head_dim
        )
        index_key = indexer.k_norm(indexer.wk(hidden_states)[0]).unsqueeze(1)
        index_query, index_key = self._rotate_indexer(positions, index_query, index_key)
        index_weights = (
            indexer.weights_proj(hidden_states.float())[0]
            * indexer.weights_softmax_scale
        )
        output = ctx.attn_backend.forward(
            query,
            None,
            None,
            self.attn_mqa,
            ctx.token_to_kv_pool,
            ctx.forward_mode,
            ctx.bs,
            fused=False,
            save_kv_cache=False,
            # The Ascend leaf publishes after writing its Index-K cache too.
            record_kv_cache=False,
            index_query=index_query,
            index_key=index_key,
            prefill_latent=None,
            index_weights=index_weights,
            index_init_tokens=self.index_init_tokens,
            index_local_tokens=self.index_local_tokens,
            head_major_output=True,
        )
        projected = torch.bmm(output, self.w_vc).transpose(0, 1).contiguous()
        return self.o_proj(projected.flatten(1))[0]


class AscendDSARuntime:
    """Prepare native DSA once before profiling; retain it across cache rebinds."""

    def __init__(self, layers, mapping):
        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager as pg,
        )

        self.layers = tuple(layers)
        if not all(supports_fused_dsa_decode(layer) for layer in self.layers):
            raise ValueError("Loaded weights do not meet the native DSA projection ABI")
        self.ranks = tuple(mapping.attn.tp_group)
        if len(self.ranks) != 8 or not self.layers:
            raise ValueError(
                "Native DSA requires a complete eight-rank attention group"
            )
        rank = self.ranks.index(dist.get_rank())
        group = pg.get_dedicated_device_group(self.ranks, "ascend_dsa")
        pg.init_process_group(self.ranks, backend="gloo")
        cpu_group = pg.get_process_group("gloo", self.ranks)
        probe = torch.ones(1, device=self.layers[0].q_a_proj.weight.device)
        dist.all_reduce(probe, group=group)
        name = group._get_backend(probe.device).get_hccl_comm_name(dist.get_rank())
        self.group = name
        self.resources = prepare_dsa_resources(group=name, rank=rank)
        self.window = self.resources.window
        self.outputs = self.resources.outputs
        self.backends = weakref.WeakSet()
        self.runtimes = []
        self.closed = False
        try:
            torch.npu.synchronize()
            dist.barrier(group=cpu_group)
            for layer in self.layers:
                runtime = FusedDSADecode(
                    layer,
                    rank=rank,
                    group=pg.get_device_process_group(self.ranks),
                    eps_q=layer.q_a_layernorm.variance_epsilon,
                    eps_kv=layer.kv_a_layernorm.variance_epsilon,
                    owner=self,
                )
                runtime.window = self.window
                layer.configure_fused_decode(runtime)
                self.runtimes.append(runtime)
        except BaseException:
            self.close()
            raise

    def bind_backend(self, backend):
        if self.closed:
            raise RuntimeError("DSA runtime is closed")
        if backend in self.backends:
            return
        pending = [backend]
        found = False
        while pending:
            node = pending.pop()
            if isinstance(node, AscendDSABackend):
                if tuple(node._dcp.ranks) != self.ranks:
                    raise ValueError("DSA model and cache backend groups differ")
                node.configure_fused_decode(
                    group=self.group,
                    ranks=self.ranks,
                    attention_outputs=self.outputs,
                )
                found = True
            pending.extend(node.child_backends())
        if not found:
            raise ValueError("Native DSA requires an Ascend DSA cache backend")
        self.backends.add(backend)

    def close(self):
        """The caller has drained forwards and destroyed every referencing graph."""
        if self.closed:
            return
        torch.npu.synchronize()
        for layer in self.layers:
            layer.configure_fused_decode(None)
        for runtime in self.runtimes:
            runtime.window = None
        self.resources.release(self.window)
        self.resources = None
        self.window = None
        self.outputs.clear()
        self.runtimes.clear()
        self.backends.clear()
        self.closed = True


__all__ = [
    "AscendDSAAttention",
    "AscendDSARuntime",
    "AscendDSAIndexer",
    "FusedDSADecode",
    "pack_qb_panels",
    "pack_down_qa_panels",
    "pack_score_fp32",
]
