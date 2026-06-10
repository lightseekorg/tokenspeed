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

"""Inference-only Nemotron-H model.

Adapted from sgl-project/sglang at commit
03c77dc33d0a051aa15c1235407440d9d107b98f
(`python/sglang/srt/models/nemotron_h.py`). Mamba2 execution uses the
TokenSpeed-kernel Triton Mamba2 kernels.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch
from torch import nn

from tokenspeed.runtime.configs.nemotron_h_config import (
    ATTENTION,
    MAMBA,
    MLP,
    MOE,
    NemotronHConfig,
)
from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention.linear.mamba2 import MambaMixer2
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.moe.layer import MoELayer
from tokenspeed.runtime.layers.moe.topk import TopK
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.utils import get_layer_id
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.utils import add_prefix, make_layers
from tokenspeed.runtime.utils.env import global_server_args_dict

_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.*\.mixer\.experts)\."
    r"(?P<expert_id>\d+)\."
    r"(?P<proj>up_proj|down_proj)\."
    r"(?P<suffix>.+)$"
)


def _relu2(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x).square()


class NemotronHMLP(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        mapping: Mapping,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        *,
        bias: bool = False,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.up_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=intermediate_size,
            bias=bias,
            quant_config=quant_config,
            tp_rank=mapping.dense.tp_rank,
            tp_size=mapping.dense.tp_size,
            tp_group=mapping.dense.tp_group,
            prefix=add_prefix("up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=config.hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            tp_rank=mapping.dense.tp_rank,
            tp_size=mapping.dense.tp_size,
            tp_group=mapping.dense.tp_group,
            prefix=add_prefix("down_proj", prefix),
        )
        if config.mlp_hidden_act != "relu2":
            raise ValueError(
                f"Unsupported Nemotron-H MLP activation: {config.mlp_hidden_act}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.up_proj(x)
        x = _relu2(x)
        x, _ = self.down_proj(x)
        return x


class NemotronHMoE(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.layer_id = layer_id
        self.routed_scaling_factor = config.routed_scaling_factor
        self.use_latent_moe = getattr(config, "moe_latent_size", None) is not None
        self.moe_hidden_size = (
            config.moe_latent_size if self.use_latent_moe else config.hidden_size
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32)
        )

        num_experts = config.n_routed_experts + (
            global_server_args_dict.get("ep_num_redundant_experts") or 0
        )
        self.experts = MoELayer(
            top_k=config.num_experts_per_tok,
            num_experts=num_experts,
            hidden_size=self.moe_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            layer_index=layer_id,
            prefix=add_prefix("experts", prefix),
            tp_rank=self.mapping.moe.tp_rank,
            tp_size=self.mapping.moe.tp_size,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
            activation=config.mlp_hidden_act,
            routing_config={
                "n_group": config.n_group,
                "topk_group": config.topk_group,
                "correction_bias": self.gate.e_score_correction_bias,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
            },
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            use_grouped_topk=True,
            topk_group=config.topk_group,
            num_expert_group=config.n_group,
            renormalize=config.norm_topk_prob,
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=1.0,
            output_format=self.experts.topk_output_format,
            apply_routed_scaling_factor_on_output=(
                self.experts.apply_routed_scaling_factor_on_output
            ),
        )

        if config.n_shared_experts:
            self.shared_experts = NemotronHMLP(
                config,
                mapping,
                intermediate_size=(
                    config.moe_shared_expert_intermediate_size * config.n_shared_experts
                ),
                quant_config=quant_config,
                bias=config.mlp_bias,
                reduce_results=True,
                prefix=add_prefix("shared_experts", prefix),
            )
        else:
            self.shared_experts = None

        if self.use_latent_moe:
            self.fc1_latent_proj = ReplicatedLinear(
                input_size=config.hidden_size,
                output_size=self.moe_hidden_size,
                bias=config.mlp_bias,
                quant_config=quant_config,
                prefix=add_prefix("fc1_latent_proj", prefix),
            )
            self.fc2_latent_proj = ReplicatedLinear(
                input_size=self.moe_hidden_size,
                output_size=config.hidden_size,
                bias=config.mlp_bias,
                quant_config=quant_config,
                prefix=add_prefix("fc2_latent_proj", prefix),
            )
        else:
            self.fc1_latent_proj = None
            self.fc2_latent_proj = None

    def forward(self, hidden_states: torch.Tensor, ctx: ForwardContext) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        router_logits, _ = self.gate(hidden_states.to(dtype=torch.float32))

        shared_output = (
            self.shared_experts(hidden_states)
            if self.shared_experts is not None
            else None
        )

        if hidden_states.shape[0] > 0:
            topk_output = self.topk(hidden_states, router_logits)
        else:
            topk_output = self.topk.empty_topk_output(
                hidden_states.device,
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

        routed_states = hidden_states
        if self.use_latent_moe:
            assert self.fc1_latent_proj is not None
            routed_states, _ = self.fc1_latent_proj(routed_states)

        final_hidden_states = self.experts(
            hidden_states=routed_states,
            topk_output=topk_output,
            num_global_tokens=ctx.input_num_tokens,
            max_num_tokens_per_gpu=ctx.input_num_tokens,
        )

        if hidden_states.dtype != torch.float16:
            final_hidden_states *= self.routed_scaling_factor
        elif shared_output is not None:
            shared_output *= 1.0 / self.routed_scaling_factor

        if self.use_latent_moe:
            assert self.fc2_latent_proj is not None
            final_hidden_states, _ = self.fc2_latent_proj(final_hidden_states)

        if self.mapping.moe.has_tp_ep:
            final_hidden_states = all_reduce(
                final_hidden_states,
                self.mapping.moe.tp_ep_group,
            )

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states.view(num_tokens, hidden_dim)


class NemotronHAttention(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.hidden_size = config.hidden_size
        self.tp_rank = mapping.attn.tp_rank
        self.tp_size = mapping.attn.tp_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            reduce_results=True,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("o_proj", prefix),
        )
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=config.sliding_window or -1,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v, ctx, out_cache_loc)
        if len(attn_output.size()) == 3:
            attn_output = attn_output.reshape(attn_output.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        return output


class NemotronHDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def _norm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)
        return hidden_states, residual


class NemotronHMLPDecoderLayer(NemotronHDecoderLayer):
    def __init__(
        self,
        config: NemotronHConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, mapping, layer_id, quant_config, prefix)
        hybrid_override_pattern = config.hybrid_override_pattern
        mlp_index = hybrid_override_pattern[: layer_id + 1].count(MLP) - 1
        if isinstance(config.intermediate_size, list):
            if len(config.intermediate_size) == 1:
                intermediate_size = config.intermediate_size[0]
            else:
                intermediate_size = config.intermediate_size[mlp_index]
        else:
            intermediate_size = config.intermediate_size
        self.mixer = NemotronHMLP(
            config,
            mapping,
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            bias=config.mlp_bias,
            prefix=add_prefix("mixer", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del ctx, out_cache_loc
        hidden_states, residual = self._norm(hidden_states, residual)
        return self.mixer(hidden_states), residual


class NemotronHMoEDecoderLayer(NemotronHDecoderLayer):
    def __init__(
        self,
        config: NemotronHConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, mapping, layer_id, quant_config, prefix)
        layer_config = config.get_nemotron_h_config_for_layer(layer_id)
        self.mixer = NemotronHMoE(
            layer_config,
            mapping=mapping,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("mixer", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del out_cache_loc
        hidden_states, residual = self._norm(hidden_states, residual)
        return self.mixer(hidden_states, ctx), residual


class NemotronHMambaDecoderLayer(NemotronHDecoderLayer):
    def __init__(
        self,
        config: NemotronHConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, mapping, layer_id, quant_config, prefix)
        self.mixer = MambaMixer2(
            config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("mixer", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del out_cache_loc
        hidden_states, residual = self._norm(hidden_states, residual)
        output = torch.empty_like(hidden_states)
        linear_backend = ctx.attn_backend.linear_attn_backend
        linear_backend.forward_mamba2(
            mixer=self.mixer,
            layer_id=self.layer_id,
            hidden_states=hidden_states,
            output=output,
        )
        return output, residual


class NemotronHAttentionDecoderLayer(NemotronHDecoderLayer):
    def __init__(
        self,
        config: NemotronHConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, mapping, layer_id, quant_config, prefix)
        layer_config = config.get_nemotron_h_config_for_layer(layer_id)
        self.mixer = NemotronHAttention(
            layer_config,
            mapping=mapping,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("mixer", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self._norm(hidden_states, residual)
        return self.mixer(hidden_states, ctx, out_cache_loc), residual


ALL_DECODER_LAYER_TYPES = {
    ATTENTION: NemotronHAttentionDecoderLayer,
    MLP: NemotronHMLPDecoderLayer,
    MAMBA: NemotronHMambaDecoderLayer,
    MOE: NemotronHMoEDecoderLayer,
}


class NemotronHModel(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )

        def get_layer(idx: int, prefix: str):
            layer_class = ALL_DECODER_LAYER_TYPES[config.hybrid_override_pattern[idx]]
            return layer_class(
                config=config,
                mapping=self.mapping,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=add_prefix("layers", prefix),
        )
        self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        del positions
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states=hidden_states,
                residual=residual,
                ctx=ctx,
                out_cache_loc=out_cache_loc,
            )

        hidden_states, _ = self.norm_f(hidden_states, residual)
        return hidden_states, None


class NemotronHForCausalLM(BaseCausalLM):
    model_cls = NemotronHModel

    remap_prefix = {"backbone.": "model."}
    remap_substr = {
        "A_log": "A",
        "embeddings": "embed_tokens",
        "k_proj.k_scale": "attn.k_scale",
        "v_proj.v_scale": "attn.v_scale",
    }

    def get_stacked_params_mapping(self) -> list[tuple[str, str, str]]:
        return [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

    def get_skip_weight_names(self) -> list[str]:
        return [
            "rotary_emb.inv_freq",
            "rotary_emb.cos_cached",
            "rotary_emb.sin_cached",
        ]

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    @classmethod
    def _remap_weight_name(cls, name: str) -> str:
        for old, new in cls.remap_prefix.items():
            if name.startswith(old):
                name = new + name[len(old) :]
        for old, new in cls.remap_substr.items():
            name = name.replace(old, new)
        return name

    def _try_load_expert_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
    ) -> bool:
        if ".mixer.experts." in name and ".gate_proj." in name:
            return True

        match = _EXPERT_WEIGHT_RE.match(name)
        if match is None:
            return False

        layer_id = get_layer_id(name)
        if layer_id is None:
            return True
        layer_config = self.config.get_nemotron_h_config_for_layer(layer_id)
        num_experts = layer_config.n_routed_experts
        num_local_experts = num_experts // self.mapping.moe.ep_size
        start_expert = num_local_experts * self.mapping.moe.ep_rank
        expert_id = int(match.group("expert_id"))
        if not (start_expert <= expert_id < start_expert + num_local_experts):
            return True

        local_expert_id = expert_id - start_expert
        shard_id = "w13" if match.group("proj") == "up_proj" else "w2"
        target_prefix = "w13_" if shard_id == "w13" else "w2_"
        target_name = f"{match.group('prefix')}.{target_prefix}{match.group('suffix')}"
        param = params_dict.get(target_name)
        if param is None:
            return True

        param.weight_loader(
            param,
            loaded_weight,
            shard_id=shard_id,
            local_expert_id=local_expert_id,
        )
        return True

    @staticmethod
    def _ensure_kv_scale_is_scalar_tensor(
        name: str, loaded_weight: torch.Tensor
    ) -> torch.Tensor:
        assert loaded_weight.numel() == 1
        assert loaded_weight.device == torch.device("cpu")
        assert loaded_weight.dtype == torch.float32
        # Must be on device before CUDA graph capture:
        loaded_weight = loaded_weight.to("cuda")
        return loaded_weight.detach().reshape(())

    def _try_load_kv_scale(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        modules_dict: dict[str, nn.Module],
    ) -> bool:
        if not (name.endswith(".attn.k_scale") or name.endswith(".attn.v_scale")):
            return False

        module_name, scale_name = name.rsplit(".", 1)
        module = modules_dict.get(module_name)
        if not isinstance(module, PagedAttention):
            return False

        scale = self._ensure_kv_scale_is_scalar_tensor(name, loaded_weight)
        setattr(module, scale_name, scale)
        setattr(module, f"{scale_name}_float", float(scale.item()))
        return True

    def has_kv_cache_scales(self) -> bool:
        attn_layers = [
            module for module in self.modules() if isinstance(module, PagedAttention)
        ]
        return bool(attn_layers) and all(
            module.k_scale is not None and module.v_scale is not None
            for module in attn_layers
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]], **kwargs):
        del kwargs
        stacked_params_mapping = self.get_stacked_params_mapping()
        skip_patterns = self.get_skip_weight_names()
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        modules_dict = dict(self.named_modules(remove_duplicate=False))

        for name, loaded_weight in weights:
            name = self._remap_weight_name(name)
            if any(pattern in name for pattern in skip_patterns):
                continue
            if "mtp" in name:
                continue

            layer_id = get_layer_id(name)
            if layer_id is not None and layer_id >= self.config.num_hidden_layers:
                continue

            if self._try_load_kv_scale(name, loaded_weight, modules_dict):
                continue

            if self._try_load_expert_weight(name, loaded_weight, params_dict):
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
                weight_loader = getattr(param, "weight_loader")
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

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class NemotronHPuzzleForCausalLM(NemotronHForCausalLM):
    pass


EntryClass = [NemotronHForCausalLM, NemotronHPuzzleForCausalLM]
