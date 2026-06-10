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

"""Inference-only dense Llama model compatible with HuggingFace weights.

Covers Llama-2 / Llama-3 / Llama-3.1 / Llama-3.2 dense checkpoints whose
``config.architectures`` is ``["LlamaForCausalLM"]``. MoE and Eagle3 draft
variants have their own modules (``longcat_large.py``, ``llama_eagle3.py``).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers import LlamaConfig

from tokenspeed.runtime.configs.utils import get_rope_theta
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.activation import SiluAndMul
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base import (
    BaseCausalLM,
    BaseDecoderLayer,
    BaseTransformerModel,
)
from tokenspeed.runtime.models.utils import create_fused_set_kv_buffer_arg
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.pdl import pdl_enabled


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_rank = mapping.dense.tp_rank
        tp_size = mapping.dense.tp_size
        tp_group = mapping.dense.tp_group

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        mapping: Mapping,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        qkv_input_size: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.attn_tp_size = mapping.attn.tp_size
        self.attn_tp_rank = mapping.attn.tp_rank
        attn_tp_group = mapping.attn.tp_group

        self.total_num_heads = num_heads
        assert self.total_num_heads % self.attn_tp_size == 0
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= self.attn_tp_size:
            assert self.total_num_kv_heads % self.attn_tp_size == 0
        else:
            assert self.attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        rope_theta = get_rope_theta(config)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # Dense Llama is consistently bias-free (`attention_bias=False` in every
        # upstream release). Still read it off the config so forks that flip
        # the flag load without surprises.
        attention_bias = getattr(config, "attention_bias", False)

        self.qkv_proj = QKVParallelLinear(
            qkv_input_size or hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            tp_group=attn_tp_group,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            tp_group=attn_tp_group,
            prefix=add_prefix("o_proj", prefix),
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        # Skip the QKV projection, RoPE, attention, and o_proj kernels when
        # the batch row is empty (e.g. idle ranks under DP attention). Matches
        # the short-circuit ``LlamaMLP.forward`` already has.
        if hidden_states.shape[0] == 0:
            return hidden_states.new_zeros(
                (0, self.hidden_size), dtype=hidden_states.dtype
            )
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self._attn(positions, q, k, v, ctx, out_cache_loc)
        output, _ = self.o_proj(attn_output)
        return output

    def _attn(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        """RoPE + attention (pre-o_proj), with optional fused KV pre-write.

        When the backend supports KV pre-write *and* ``create_fused_set_kv_buffer_arg``
        accepts the layer's scales, fused rope writes KV directly into the cache
        so the attention call can run with ``save_kv_cache=False`` (saves one
        kernel launch). Otherwise we fall back to plain RoPE + ``self.attn(q, k, v)``
        so the backend writes KV the normal way — without this fallback, layers
        with non-trivial k/v scales silently lose their KV writes. Subclasses
        (e.g. Eagle3 draft head) override this hook to insert spec-decode
        behaviour around the same scaffolding.
        """
        if ctx.attn_backend.support_kv_cache_prewrite(ctx.forward_mode):
            fused_kv_arg = self._build_fused_kv_arg(v, ctx, out_cache_loc)
            if fused_kv_arg is not None:
                q_rope = self._fused_rope_kv_write(positions, q, k, fused_kv_arg)
                return self.attn(
                    q_rope,
                    None,
                    None,
                    save_kv_cache=False,
                    ctx=ctx,
                    out_cache_loc=out_cache_loc,
                )
        q, k = self.rotary_emb(positions, q, k)
        return self.attn(q, k, v, ctx=ctx, out_cache_loc=out_cache_loc)

    def _build_fused_kv_arg(
        self,
        v: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ):
        """Try to build the fused RoPE+KV-write descriptor; returns ``None`` if
        the helper rejects the layer (e.g. non-trivial k/v scales)."""
        n = v.shape[0]
        return create_fused_set_kv_buffer_arg(
            value=v.view(n, self.num_kv_heads, self.head_dim),
            layer=self.attn,
            out_cache_loc=out_cache_loc,
            token_to_kv_pool=ctx.token_to_kv_pool,
        )

    def _fused_rope_kv_write(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        fused_kv_arg,
    ) -> torch.Tensor:
        """Fused RoPE that writes KV into cache (via ``fused_kv_arg``) and
        returns the rope'd Q."""
        n = q.shape[0]
        q_rope = torch.empty((n, self.q_size), dtype=q.dtype, device=q.device)
        self.rotary_emb(
            positions,
            q,
            k,
            fused_set_kv_buffer_arg=fused_kv_arg,
            output_q_rope=q_rope,
            enable_pdl=pdl_enabled(),
        )
        return q_rope


class LlamaDecoderLayer(BaseDecoderLayer):

    def resolve_attn(self, prefix: str) -> nn.Module:
        return LlamaAttention(
            config=self.config,
            mapping=self.mapping,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            layer_id=self.layer_id,
            quant_config=self.quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

    def resolve_mlp(self, prefix: str) -> nn.Module:
        return LlamaMLP(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
            mapping=self.mapping,
            quant_config=self.quant_config,
            prefix=add_prefix("mlp", prefix),
        )


class LlamaModel(BaseTransformerModel):
    layer_cls = LlamaDecoderLayer


class LlamaForCausalLM(BaseCausalLM):
    model_cls = LlamaModel

    # BitsAndBytes target/stacked modules — kept in sync with the Qwen3 / MoE
    # variants so a single quantization config works across the Llama family.
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def get_stacked_params_mapping(self) -> list[tuple[str, str, int | str]]:
        return [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], **kwargs: Any
    ) -> None:
        stacked_params_mapping = self.get_stacked_params_mapping()
        params_dict = dict(self.named_parameters())
        tie_word_embeddings = getattr(self.config, "tie_word_embeddings", False)

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            # Llama-3.2-1B / 3B ship with tied input+output embeddings — some HF
            # checkpoint variants still serialize lm_head.weight, skip it so we
            # don't double-load into the shared embed_tokens parameter.
            if tie_word_embeddings and "lm_head.weight" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                # Fused q/k/v and gate/up parameters are built by distributed
                # linear layers that install ``weight_loader`` during init; the
                # ``getattr`` fallback just guards against stray non-fused
                # parameters that happened to match the pattern (e.g. a user
                # fork that registers a plain ``qkv_proj`` buffer).
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = LlamaForCausalLM
