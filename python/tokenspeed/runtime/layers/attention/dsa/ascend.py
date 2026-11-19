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

import torch
from tokenspeed_kernel.ops.attention.prologue import latent_store
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_ops import all_gather
from tokenspeed.runtime.layers.attention.backends.paged.ascend_dsa_tp_kvp_decode import (
    DSA_DECODE_ROWS,
)
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.model_loader.weight_utils import bind_or_copy
from tokenspeed.runtime.models.deepseek_v3 import (
    _prepare_mla_kv_b_proj_weights,
)
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.env import global_server_args_dict


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
        self.rotary_emb = None
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

    def forward(self, hidden_states, q_lora):
        """Project Index-Q/K and scores from the unscaled normalized Q latent."""
        query = self.wq_b(q_lora)[0].view(-1, self.index_n_heads, self.index_head_dim)
        key = self.k_norm(self.wk(hidden_states)[0]).unsqueeze(1)
        weights = (
            self.weights_proj(hidden_states.float())[0] * self.weights_softmax_scale
        )
        return query, key, weights


class AscendDSAAttention(nn.Module):
    """DSA projections and attention with one independent indexer per layer."""

    supports_head_tp = False

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
        quant_config: QuantizationConfig | None = None,
        layer_id: int | None = None,
        prefix: str = "",
        reduce_attn_results: bool = True,
        *,
        lora_norm_eps: float,
        prefill_enabled: bool,
        decode_enabled: bool | None = None,
    ) -> None:
        if config.index_n_heads not in (16, 32):
            raise NotImplementedError("Ascend DSA supports only Indexer16/32")
        if not config.mla_use_nope or config.index_use_rope:
            raise ValueError(
                "Ascend DSA requires both attention and Indexer RoPE disabled"
            )
        self.use_output_gate = bool(getattr(config, "mla_use_output_gate", False))
        self.scale_q_lora = bool(getattr(config, "mla_scale_q_lora", False))
        self.scale_kv_lora = bool(getattr(config, "mla_scale_kv_lora", False))
        if self.use_output_gate:
            raise ValueError("Native DSA requires no output gate")
        super().__init__()
        if mapping.attn.has_head_tp:
            raise NotImplementedError(
                "Ascend DSA does not yet implement the head-TP token exchange"
            )
        if global_server_args_dict["tp_batch_invariant"] != "none":
            raise ValueError("--tp-batch-invariant requires supported head-TP exchange")
        if q_lora_rank is None:
            raise ValueError("Ascend DSA requires q_lora_rank")
        self.config, self.mapping = config, mapping
        self.layer_id, self.hidden_size = layer_id, hidden_size
        self.qk_nope_head_dim, self.qk_rope_head_dim = (
            qk_nope_head_dim,
            qk_rope_head_dim,
        )
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank, self.kv_lora_rank = q_lora_rank, kv_lora_rank
        self.num_heads = num_heads
        # Earlier callers select only Prefill capability. Their full projections
        # also serve local (KVP1) Decode. Sharded Decode must be requested on
        # replicas that serve both phases.
        if decode_enabled is None:
            decode_enabled = not prefill_enabled or mapping.attn.kvp_size == 1
        use_decode_shards = not prefill_enabled or (
            decode_enabled and mapping.attn.kvp_size > 1
        )
        self.head_tp_size = mapping.attn.head_tp_size if use_decode_shards else 1
        self.head_tp_rank = mapping.attn.head_tp_rank if use_decode_shards else 0
        self.head_tp_group = mapping.attn.head_tp_group if use_decode_shards else None
        if decode_enabled and mapping.attn.kvp_size == 1 and self.head_tp_size != 1:
            raise ValueError("Local DSA Decode requires TP1/KVP1 projections")
        if num_heads % self.head_tp_size:
            raise ValueError(
                f"num_heads={num_heads} must be divisible by head TP size {self.head_tp_size}"
            )
        self.num_local_heads = num_heads // self.head_tp_size
        self.scaling = self.qk_head_dim**-0.5

        # Construct only the checkpoint projections this layer actually uses.
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
            tp_rank=self.head_tp_rank,
            tp_size=self.head_tp_size,
            tp_group=self.head_tp_group,
            prefix=add_prefix("q_b_proj", prefix),
        )
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
            tp_rank=self.head_tp_rank,
            tp_size=self.head_tp_size,
            tp_group=self.head_tp_group,
        )
        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            reduce_results=reduce_attn_results,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=self.head_tp_rank,
            tp_size=self.head_tp_size,
            tp_group=self.head_tp_group,
        )
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=lora_norm_eps)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=lora_norm_eps)
        self.rotary_emb = None
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
        self.w_kc = None
        self.w_vc = None
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
        self._fused_tp_kvp_decode_runtime = None
        self.prefill_enabled = prefill_enabled
        self.decode_enabled = decode_enabled
        self._prefill_weights: tuple[torch.Tensor, ...] | None = None

    def configure_fused_tp_kvp_decode(self, runtime) -> None:
        """Attach prepared weights and communication state outside forward."""
        self._fused_tp_kvp_decode_runtime = runtime

    def _prepare_lora_scales(self):
        """Fold native KV scaling; numerical variants may keep it in forward."""
        # The Indexer consumes unscaled normalized Q latent, so Q scaling
        # stays after the main projection and never enters the shared norm.
        if self.scale_kv_lora:
            self.kv_a_layernorm.weight.data.mul_(
                math.sqrt(self.hidden_size / self.kv_lora_rank)
            )

    def process_weights_after_loading(self, _module=None):
        if not self._scales_prepared:
            self._prepare_lora_scales()
            self._scales_prepared = True

        self.w_kc, self.w_vc = _prepare_mla_kv_b_proj_weights(
            self.kv_b_proj.weight, self
        )
        if not self.prefill_enabled:
            return

        local = (self.q_b_proj.weight, self.w_kc, self.w_vc, self.o_proj.weight)
        if self.head_tp_size == 1:
            # Keep the original absorbed strides and avoid redundant copies.
            self._prefill_weights = local
            return

        # QB, KC and VC are partitioned by head; O is partitioned by input.
        # Reuse each prepared buffer when the model reloads its coefficients.
        previous = self._prefill_weights or (None, None, None, None)
        self._prefill_weights = tuple(
            bind_or_copy(
                buffer, all_gather(weight.contiguous(), self.head_tp_group, dim)
            )
            for buffer, weight, dim in zip(previous, local, (0, 0, 0, 1), strict=True)
        )

    def _forward_fused_tp_kvp_decode(self, positions, hidden_states, ctx, comm_manager):
        packed_query, index_query, index_weights = (
            self._fused_tp_kvp_decode_runtime.prolog(hidden_states, positions, ctx)
        )
        out = ctx.attn_backend.forward(
            packed_query[self._fused_tp_kvp_decode_runtime.rank],
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
        return self._fused_tp_kvp_decode_runtime.output_projection(out)

    def _forward_prefill(self, positions, hidden, ctx):
        if self._prefill_weights is None:
            raise RuntimeError("Prepare token-local DSA Prefill weights before forward")
        qb, kc, vc, wo = self._prefill_weights
        indexer = self.indexer
        if hidden.shape[0]:
            q_lora = self.q_a_layernorm(self.q_a_proj(hidden)[0])
            iq, ik, weights = indexer(hidden, q_lora)
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
            if ctx.token_to_kv_pool.latent_write_sanitizes:
                latent = torch.nan_to_num(latent)
        else:
            iq = hidden.new_empty((0, indexer.index_n_heads, indexer.index_head_dim))
            ik = hidden.new_empty((0, 1, indexer.index_head_dim))
            weights = torch.empty(
                (0, indexer.index_n_heads), dtype=torch.float32, device=hidden.device
            )
            query = hidden.new_empty(
                (0, kc.shape[0], self.kv_lora_rank + self.qk_rope_head_dim)
            )
            latent = hidden.new_empty((0, self.kv_lora_rank + self.qk_rope_head_dim))
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
            index_query=iq,
            prefill_latent=latent.contiguous(),
            index_key=ik.contiguous(),
            index_weights=weights,
            index_init_tokens=self.index_init_tokens,
            index_local_tokens=self.index_local_tokens,
            head_major_output=True,
        )
        if hidden.shape[0] == 0:
            return hidden
        projected = torch.bmm(output, vc).transpose(0, 1).contiguous()
        return F.linear(projected.flatten(1), wo)

    def forward(self, positions, hidden_states, ctx, comm_manager):
        if ctx.forward_mode.is_extend_or_mixed():
            if not self.prefill_enabled:
                raise RuntimeError(
                    "DSA Prefill weights were disabled for this decode-only model"
                )
            return self._forward_prefill(positions, hidden_states, ctx)
        if not self.decode_enabled:
            raise RuntimeError(
                "DSA Decode weights were disabled for this Prefill-only model"
            )
        if hidden_states.shape[0] == 0:
            return hidden_states
        if self.head_tp_size == 1 and self.mapping.attn.kvp_size == 1:
            return self._forward_decode(hidden_states, ctx)
        return self._forward_tp_kvp_decode(positions, hidden_states, ctx, comm_manager)

    def _forward_tp_kvp_decode(self, positions, hidden_states, ctx, comm_manager):
        if (
            self._fused_tp_kvp_decode_runtime is not None
            and hidden_states.ndim == 2
            and hidden_states.shape[0] in DSA_DECODE_ROWS
            and hidden_states.shape[1] == self._fused_tp_kvp_decode_runtime.hidden_size
            and ctx.forward_mode.is_decode()
            and ctx.num_extends == 0
            and 1 <= ctx.bs <= 64
            and ctx.output_layout.decode_width in (1, 8)
            and hidden_states.shape[0] == ctx.bs * ctx.output_layout.decode_width
        ):
            return self._forward_fused_tp_kvp_decode(
                positions, hidden_states, ctx, comm_manager
            )
        return self._forward_decode(hidden_states, ctx)

    def _forward_decode(self, hidden_states, ctx):
        """Ordinary projections and owner-local stores; the backend owns attention."""
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
        latent_store(
            latent_cache,
            kv_lora_rank=self.kv_lora_rank,
            cache=self.attn_mqa._local_target(slots, ctx),
        )
        index_query, index_key, index_weights = self.indexer(hidden_states, q_lora)
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


__all__ = ["AscendDSAAttention", "AscendDSAIndexer"]
