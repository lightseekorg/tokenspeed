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

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn

from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.activation import SiluAndMul
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.env import global_server_args_dict


class DFlashAttention(nn.Module):
    def __init__(
        self,
        config,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.hidden_size = int(config.hidden_size)
        self.tp_rank = self.mapping.attn.tp_rank
        self.tp_size = self.mapping.attn.tp_size
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(
            getattr(config, "num_key_value_heads", self.total_num_heads)
        )
        assert self.total_num_heads % self.tp_size == 0
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0

        self.num_heads = self.total_num_heads // self.tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = int(
            getattr(config, "head_dim", self.hidden_size // self.total_num_heads)
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bool(getattr(config, "attention_bias", False)),
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=bool(getattr(config, "attention_bias", False)),
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            reduce_results=False,
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=int(getattr(config, "max_position_embeddings", 32768)),
            base=float(getattr(config, "rope_theta", 1000000)),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        # The FA4 MHA extend selector currently has no sliding-window kernel
        # for this draft shape. Use full attention for draft proposals; target
        # verification remains authoritative for accepted tokens.
        sliding_window = -1
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window,
        )
        self.attn.non_causal = True

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = q.reshape(-1, self.head_dim)
        k = k.reshape(-1, self.head_dim)
        q = self.q_norm(q).view(-1, self.q_size)
        k = self.k_norm(k).view(-1, self.kv_size)
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        k_cache = k.view(-1, self.num_kv_heads, self.head_dim)
        v_cache = v.view(-1, self.num_kv_heads, self.head_dim)
        ctx.token_to_kv_pool.set_kv_buffer(
            self.attn,
            out_cache_loc,
            k_cache,
            v_cache,
            self.attn.k_scale,
            self.attn.v_scale,
        )
        attn_output = self.attn(
            q,
            None,
            None,
            ctx,
            out_cache_loc,
            save_kv_cache=False,
        )
        if len(attn_output.size()) == 3:
            attn_output = attn_output.reshape(attn_output.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        return output

    def kv_proj_only(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qkv, _ = self.qkv_proj(hidden_states)
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return k, v

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        k_shape = k.shape
        return self.k_norm(k.reshape(-1, self.head_dim)).view(k_shape)

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        dummy_q = k.new_empty(k.shape)
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k


class DFlashMLP(nn.Module):
    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=mapping.dense.tp_rank,
            tp_size=mapping.dense.tp_size,
            tp_group=mapping.dense.tp_group,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            reduce_results=False,
            tp_rank=mapping.dense.tp_rank,
            tp_size=mapping.dense.tp_size,
            tp_group=mapping.dense.tp_group,
        )
        if getattr(config, "hidden_act", "silu") != "silu":
            raise ValueError("DFlash only supports silu activation.")
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DFlashDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.mapping = mapping
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.self_attn = DFlashAttention(
            config=config,
            mapping=mapping,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.mlp = DFlashMLP(
            config=config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ctx.forward_mode.is_idle():
            hidden_states = self.mlp(hidden_states)
            return hidden_states, residual

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif (
            ctx.input_num_tokens > global_server_args_dict["comm_fusion_max_num_tokens"]
        ):
            hidden_states = all_reduce(hidden_states, self.mapping.dense.tp_group)
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            hidden_states, residual, *_ = (
                self.input_layernorm.forward_with_allreduce_fusion(
                    self.mapping.dense.tp_rank,
                    self.mapping.dense.tp_group,
                    hidden_states,
                    residual,
                )
            )

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
        )

        if ctx.input_num_tokens > global_server_args_dict["comm_fusion_max_num_tokens"]:
            hidden_states = all_reduce(hidden_states, self.mapping.attn.tp_group)
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        else:
            hidden_states, residual, *_ = (
                self.post_attention_layernorm.forward_with_allreduce_fusion(
                    self.mapping.attn.tp_rank,
                    self.mapping.attn.tp_group,
                    hidden_states,
                    residual,
                )
            )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DFlashDraftModel(nn.Module):
    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.layers = nn.ModuleList(
            [
                DFlashDecoderLayer(
                    config=config,
                    mapping=mapping,
                    layer_id=i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(int(config.num_hidden_layers))
            ]
        )
        self.norm = RMSNorm(int(config.hidden_size), eps=eps)
        target_layer_ids = (getattr(config, "dflash_config", {}) or {}).get(
            "target_layer_ids", []
        )
        self.num_context_features = len(target_layer_ids)
        self.fc = nn.Linear(
            self.num_context_features * int(config.hidden_size),
            int(config.hidden_size),
            bias=False,
        )
        self.hidden_norm = RMSNorm(int(config.hidden_size), eps=eps)
        self.block_size = int(getattr(config, "block_size", 8))

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm(self.fc(target_hidden))

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            if not ctx.forward_mode.is_idle():
                raise ValueError("DFlashDraftModel requires input_embeds.")
            hidden_states = self.fc.weight.new_empty((0, int(self.config.hidden_size)))
        else:
            hidden_states = input_embeds
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                ctx=ctx,
                out_cache_loc=out_cache_loc,
                residual=residual,
            )

        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)

        return LogitsProcessorOutput(
            next_token_logits=None, hidden_states=hidden_states
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())

        def resolve_name(name: str) -> str | None:
            if name in params_dict:
                return name
            if name.startswith("model.") and name[len("model.") :] in params_dict:
                return name[len("model.") :]
            return None

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                resolved = resolve_name(name.replace(weight_name, param_name))
                if resolved is None:
                    continue
                param = params_dict[resolved]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                resolved = resolve_name(name)
                if resolved is None:
                    continue
                param = params_dict[resolved]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [DFlashDraftModel]
