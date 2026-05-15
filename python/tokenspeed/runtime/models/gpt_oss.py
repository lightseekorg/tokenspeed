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

"""Inference-only GptOss model compatible with HuggingFace weights."""

# ruff: noqa: E402

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.configs.utils import get_rope_theta
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.moe.checkpoint import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.layer import MoELayer
from tokenspeed.runtime.layers.moe.topk import TopK
from tokenspeed.runtime.layers.moe.utils import get_all2all_backend
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base import (
    BaseCausalLM,
    BaseTransformerModel,
    CompiledMoEDecoderLayer,
)
from tokenspeed.runtime.models.utils import create_fused_set_kv_buffer_arg
from tokenspeed.runtime.utils import add_prefix, get_colorful_logger
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.pdl import pdl_enabled

logger = get_colorful_logger(__name__)


from tokenspeed_kernel.ops.gemm.flashinfer import tinygemm_bf16
from tokenspeed_kernel.registry import error_fn


class TinyGemmLinear(ReplicatedLinear):
    """ReplicatedLinear with a FlashInfer tinygemm BF16 fast path for small batches."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_tinygemm = (
            tinygemm_bf16 is not error_fn
            and not self.skip_bias_add
            and self.weight.is_contiguous()
            and self.weight.shape[0] % 16 == 0
            and self.weight.shape[1] % 64 == 0
            and self.weight.dtype == torch.bfloat16
            and (
                self.bias is None
                or (
                    self.bias.dtype == torch.bfloat16
                    and self.bias.is_contiguous()
                    and self.bias.shape[0] == self.weight.shape[0]
                )
            )
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (
            self._use_tinygemm
            and x.ndim == 2
            and x.is_cuda
            and x.shape[0] <= 128
            and x.is_contiguous()
            and x.shape[1] == self.weight.shape[1]
            and x.dtype == torch.bfloat16
        ):
            out = x.new_empty((x.shape[0], self.output_size))
            tinygemm_bf16(x, self.weight, out, self.bias, use_pdl=pdl_enabled())
            return out, None

        return super().forward(x)


class GptOssAttention(nn.Module):
    def __init__(
        self,
        config,
        mapping: Mapping,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        sliding_window_size: int = -1,
        layer_type: str = "",
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> None:

        super().__init__()
        self.mapping = mapping
        self.hidden_size = hidden_size
        self.sliding_window_size = sliding_window_size

        attn_tp_rank = self.mapping.attn.tp_rank
        attn_tp_size = self.mapping.attn.tp_size
        attn_tp_group = self.mapping.attn.tp_group

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = self.mapping.rank

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            params_dtype=params_dtype,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            tp_group=attn_tp_group,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.sinks = nn.Parameter(
            torch.empty(self.num_heads, dtype=torch.bfloat16), requires_grad=False
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            tp_group=attn_tp_group,
            reduce_results=False,
            params_dtype=params_dtype,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        assert layer_type in {"sliding_attention", "full_attention"}
        use_sliding_window = layer_type == "sliding_attention"
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=(sliding_window_size if use_sliding_window else -1),
        )
        self.layer_id = layer_id

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ):

        if hidden_states.shape[0] == 0:
            return hidden_states, ctx, out_cache_loc, None
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        fused_kv_arg = None
        if ctx.attn_backend.support_kv_cache_prewrite:
            n = q.shape[0]
            v_3d = v.view(n, self.num_kv_heads, self.head_dim)
            fused_kv_arg = create_fused_set_kv_buffer_arg(
                value=v_3d,
                layer=self.attn,
                out_cache_loc=out_cache_loc,
                token_to_kv_pool=ctx.token_to_kv_pool,
            )

        if fused_kv_arg is not None:
            n = q.shape[0]
            q_rope = torch.empty((n, self.q_size), dtype=q.dtype, device=q.device)
            q, k = self.rotary_emb(
                positions,
                q,
                k,
                fused_set_kv_buffer_arg=fused_kv_arg,
                output_q_rope=q_rope,
                enable_pdl=pdl_enabled(),
            )
            inner_state = q_rope, None, None
        else:
            q, k = self.rotary_emb(positions, q, k)
            inner_state = q, k, v
        return None, ctx, out_cache_loc, inner_state

    def forward_core(self, intermediate_state):

        hidden_states, ctx, out_cache_loc, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states
        # Cache was already written by the fused RoPE+KV kernel iff we took that path,
        # which is exactly when k is None in inner_state.
        save_kv_cache = inner_state[1] is not None
        attn_output = self.attn(
            *inner_state,
            save_kv_cache=save_kv_cache,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
            sinks=self.sinks,
        )
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:

        s = self.forward_prepare(
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
        )
        return self.forward_core(s)


def routing_function(hidden_states, gating_output, topk, renormalize):

    experts = torch.topk(gating_output, k=topk, dim=-1, sorted=True)
    expert_weights = torch.nn.functional.softmax(
        experts.values.to(torch.float32), dim=1
    )
    expert_indices = experts.indices.to(torch.int32)
    return expert_weights, expert_indices


class GptOssSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config,
        mapping: Mapping,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        layer_index: int = -1,
        prefix: str = "",
    ):

        super().__init__()
        self.mapping = mapping
        self.layer_index = layer_index
        self.tp_size = self.mapping.world_size
        self.hidden_size = hidden_size
        self.activation = config.hidden_act
        self.activation_alpha = getattr(config, "hidden_act_alpha", 1.702)
        self.swiglu_limit = config.swiglu_limit
        self.num_experts = (
            num_experts + global_server_args_dict["ep_num_redundant_experts"]
        )
        self.quant_config = quant_config
        if self.tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_local_experts}."
            )

        self.experts = MoELayer(
            top_k=top_k,
            num_experts=self.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=self.quant_config,
            layer_index=self.layer_index,
            prefix=add_prefix("experts", prefix),
            tp_rank=self.mapping.moe.tp_rank,
            tp_size=self.mapping.moe.tp_size,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
            activation="swiglu",
            activation_alpha=self.activation_alpha,
            swiglu_limit=self.swiglu_limit,
            # HF gpt-oss stores ``gate_up_proj_blocks`` row-interleaved
            # ([w1_0, w3_0, w1_1, w3_1, ...]) and uses the gpt-oss SwiGLU+1
            # activation silu(α·gate)·(up + 1).
            swiglu_beta=1.0,
            w13_input_layout="interleaved",
            with_bias=True,
        )

        self.router = TinyGemmLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=True,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
            params_dtype=config.torch_dtype,
        )
        self.topk = TopK(
            top_k=top_k,
            custom_routing_function=routing_function,
            output_format=self.experts.topk_output_format,
            apply_routed_scaling_factor_on_output=(
                self.experts.apply_routed_scaling_factor_on_output
            ),
            topk_indices_dtype=(
                torch.int64 if get_all2all_backend().is_deepep() else torch.int32
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:

        # router_logits: (num_tokens, n_experts)
        if hidden_states.shape[0] == 0:
            router_logits = hidden_states.new_empty(0, self.router.weight.shape[0])
        else:
            router_output = self.router(hidden_states)
            router_logits = (
                router_output[0] if isinstance(router_output, tuple) else router_output
            )
        if hidden_states.shape[0] > 0:
            topk_output = self.topk(hidden_states, router_logits)
        else:
            topk_output = self.topk.empty_topk_output(
                hidden_states.device,
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        return self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
        )

    def get_moe_weights(self) -> list[torch.Tensor]:

        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]


class _WeightCreator:
    def __init__(self, fn):
        self._fn = fn

    @staticmethod
    def maybe_materialize(obj):
        if isinstance(obj, _WeightCreator):
            output = obj._fn()
            obj._fn = None
            return output
        return obj


class GptOssConfig(PretrainedConfig):
    model_type = "gpt_oss"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def get_attention_sliding_window_size(config):
    # Aligned with HF's implementation, using sliding window inclusive with the last token
    # TokenSpeed assumes exclusive
    return config.sliding_window - 1


class GptOssDecoderLayer(CompiledMoEDecoderLayer):

    def __init__(
        self,
        config: GptOssConfig,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        sliding_window_size: int | None = None,
    ) -> None:

        self._config = config
        self._mapping = mapping
        self._quant_config = quant_config
        self._prefix = prefix

        if sliding_window_size is None:
            self.sliding_window_size = get_attention_sliding_window_size(config)
        else:
            self.sliding_window_size = sliding_window_size

        super().__init__(
            config=config,
            layer_id=layer_id,
            mapping=mapping,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.attn_tp_group = pg_manager.get_process_group(
            "nccl", self.mapping.attn.tp_group
        )
        self.attn_tp_size = self.mapping.attn.tp_size
        self.attn_tp_rank = self.mapping.attn.tp_rank

    def resolve_attn(self, prefix: str) -> nn.Module:

        config = self._config
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        return GptOssAttention(
            config=config,
            mapping=self._mapping,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=self.layer_id,
            rope_theta=get_rope_theta(config),
            rope_scaling=getattr(config, "rope_scaling", None),
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            head_dim=head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            quant_config=self._quant_config,
            prefix=add_prefix("self_attn", prefix),
            sliding_window_size=self.sliding_window_size,
            layer_type=config.layer_types[self.layer_id],
            params_dtype=config.torch_dtype,
        )

    def resolve_mlp(self, prefix: str) -> nn.Module:

        config = self._config

        return GptOssSparseMoeBlock(
            config=config,
            mapping=self._mapping,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=self._quant_config,
            layer_index=self.layer_id,
            prefix=add_prefix("mlp", prefix),
        )


class GptOssModel(BaseTransformerModel):
    layer_cls = GptOssDecoderLayer


class GptOssForCausalLM(BaseCausalLM):
    model_cls = GptOssModel
    fall_back_to_pt_during_load = False

    def get_attention_sliding_window_size(self):
        return get_attention_sliding_window_size(self.config)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        from tokenspeed.runtime.moe.expert_location import (
            ModelConfigForExpertLocation,
        )

        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_local_experts,
            num_groups=None,
        )

    def _get_default_weight_mapping(self):

        weight_mapping = {}
        weight_mapping["embedding.weight"] = "model.embed_tokens.weight"
        weight_mapping["unembedding.weight"] = "lm_head.weight"
        weight_mapping["norm.scale"] = "model.norm.weight"

        for layer_id in range(self.config.num_hidden_layers):
            pfx = f"model.layers.{layer_id}"
            bpfx = f"block.{layer_id}"

            for proj in ("q_proj", "k_proj", "v_proj"):
                weight_mapping[f"{bpfx}.attn.{proj}.weight"] = (
                    f"{pfx}.self_attn.{proj}.weight"
                )
                weight_mapping[f"{bpfx}.attn.{proj}.bias"] = (
                    f"{pfx}.self_attn.{proj}.bias"
                )

            weight_mapping[f"{bpfx}.attn.out.weight"] = f"{pfx}.self_attn.o_proj.weight"
            weight_mapping[f"{bpfx}.attn.out.bias"] = f"{pfx}.self_attn.o_proj.bias"
            weight_mapping[f"{bpfx}.attn.sinks"] = f"{pfx}.self_attn.sinks"
            weight_mapping[f"{bpfx}.attn.norm.scale"] = f"{pfx}.input_layernorm.weight"

            weight_mapping[f"{bpfx}.mlp.gate.weight"] = f"{pfx}.mlp.router.weight"
            weight_mapping[f"{bpfx}.mlp.gate.bias"] = f"{pfx}.mlp.router.bias"
            weight_mapping[f"{bpfx}.mlp.norm.scale"] = (
                f"{pfx}.post_attention_layernorm.weight"
            )
            weight_mapping[f"{bpfx}.mlp.experts.gate_up_proj"] = (
                f"{pfx}.mlp.experts.gate_up_proj"
            )
            weight_mapping[f"{bpfx}.mlp.gate_up_proj_bias"] = (
                f"{pfx}.mlp.experts.gate_up_proj_bias"
            )
            weight_mapping[f"{bpfx}.mlp.down_proj"] = f"{pfx}.mlp.experts.mlp2_weight"
            weight_mapping[f"{bpfx}.mlp.down_proj_bias"] = (
                f"{pfx}.mlp.experts.mlp2_bias"
            )

        return weight_mapping

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        is_nextn: bool = False,
        weight_name_mapping: dict = None,
    ):

        quant_config_name = (
            self.quant_config.get_name() if self.quant_config is not None else None
        )
        assert not is_nextn

        if quant_config_name == "mxfp4":
            self._load_mxfp4_weights(weights, weight_name_mapping=weight_name_mapping)
        else:
            self._load_normal_weights(weights, weight_name_mapping=weight_name_mapping)

    def _load_normal_weights(
        self,
        weights,
        weight_name_mapping: dict = None,
        other_loaded_param_names: set = None,
    ):

        attn_tp_rank = self.mapping.attn.tp_rank
        rank = self.mapping.rank
        weights = sorted(weights, key=lambda x: x[0])

        if weight_name_mapping is None:
            weight_name_mapping = self._get_default_weight_mapping()
        else:
            default_mapping = self._get_default_weight_mapping()
            default_mapping.update(weight_name_mapping)
            weight_name_mapping = default_mapping

        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        # MoE expert weights, scales, and activation scales are handled
        # by the checkpoint loader.
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params_dict,
            fused_schema=ExpertCheckpointSchema(
                gate_up_fused_name="gate_up_proj",
                down_proj_name="down_proj",
                extra_names={
                    "gate_up_bias": "gate_up_proj_bias",
                    "down_bias": "down_proj_bias",
                },
            ),
            num_experts=self.config.num_local_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
            fused_gate_up_as_w13=True,
            include_bias=True,
            fused_load_style="local_tensor",
            transpose_local_tensor_non_bias=True,
        )
        params_checker = {k: False for k in params_dict}

        for name, loaded_weight in weights:
            loaded_weight = _WeightCreator.maybe_materialize(loaded_weight)

            if weight_name_mapping and name in weight_name_mapping:
                name = weight_name_mapping[name]

            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                params_checker[name] = True
                break

            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if moe_loader.matches(name):
                    mapped_name = moe_loader.load(name, loaded_weight)
                    params_checker[mapped_name] = True
                    name = mapped_name
                else:
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    if "sinks" in name:
                        start = attn_tp_rank * param.numel()
                        param.data.copy_(loaded_weight[start : start + param.numel()])
                    else:
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    params_checker[name] = True

        not_loaded_params = []
        already_loaded = other_loaded_param_names or set()
        for k, v in params_checker.items():
            if (
                not v
                and ("weight_scale" not in k)
                and ("input_scale" not in k)
                and k not in already_loaded
            ):
                not_loaded_params.append(k)

        if rank == 0:
            if len(not_loaded_params) > 0:
                raise Exception(f"Not all parameters loaded: {not_loaded_params=}")
            else:
                logger.info("All parameters loaded successfully.")

        self.routed_experts_weights_of_layer = {
            layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
            for layer_id in range(len(self.model.layers))
        }

    def _load_mxfp4_weights(self, weights, weight_name_mapping: dict):

        mxfp4_weights = []
        normal_weights = []

        for name, weight in weights:
            if ".experts" in name:
                mxfp4_weights.append((name, weight))
            else:
                normal_weights.append((name, weight))

        mxfp4_loaded_params = self._load_mxfp4_experts_weights(mxfp4_weights)
        self._load_normal_weights(
            normal_weights,
            weight_name_mapping=weight_name_mapping,
            other_loaded_param_names=mxfp4_loaded_params,
        )

    def _load_mxfp4_experts_weights(self, weights):

        params_dict = dict(self.named_parameters())
        loaded_params: set = set()
        mxfp4_block = 32

        moe_tp_rank = self.mapping.moe.tp_rank
        moe_tp_size = self.mapping.moe.tp_size
        moe_ep_rank = self.mapping.moe.ep_rank
        moe_ep_size = self.mapping.moe.ep_size

        intermediate_size = self.config.intermediate_size
        intermediate_size_block = intermediate_size // mxfp4_block
        per_rank_intermediate_size_block = math.ceil(
            intermediate_size_block / moe_tp_size
        )
        per_rank_intermediate_size = per_rank_intermediate_size_block * mxfp4_block

        moe_num_global_experts = self.config.num_local_experts
        moe_num_local_experts = moe_num_global_experts // moe_ep_size

        moe_tp_rank_start = moe_tp_rank * per_rank_intermediate_size
        moe_tp_rank_end = min(
            (moe_tp_rank + 1) * per_rank_intermediate_size, intermediate_size
        )

        moe_ep_rank_start = moe_ep_rank * moe_num_local_experts
        moe_ep_rank_end = (moe_ep_rank + 1) * moe_num_local_experts

        def _copy_into_param(param, narrow_weight):
            if param.shape == narrow_weight.shape:
                param.data.copy_(narrow_weight)
            else:
                slices = tuple(
                    slice(0, min(p, n))
                    for p, n in zip(param.shape, narrow_weight.shape)
                )
                param.data[slices].copy_(narrow_weight[slices])

        # Detect AMD-Quark per-expert checkpoints (e.g.
        # ``amd/gpt-oss-120b-w-mxfp4-a-fp8``). These store one set of tensors
        # per expert (``...experts.{e}.gate_up_proj.{weight,...}``) plus a
        # scalar ``input_scale`` for static FP8 activation quantization.
        if any(
            re.search(r"\.experts\.\d+\.(gate_up_proj|down_proj)\.", n)
            for n, _ in weights
        ):
            return self._load_mxfp4_per_expert_weights(
                weights,
                params_dict=params_dict,
                moe_tp_rank_start=moe_tp_rank_start,
                moe_tp_rank_end=moe_tp_rank_end,
                moe_ep_rank_start=moe_ep_rank_start,
                moe_ep_rank_end=moe_ep_rank_end,
                moe_tp_rank=moe_tp_rank,
                copy_into_param=_copy_into_param,
                mxfp4_block=mxfp4_block,
            )

        for name, weight in weights:
            weight = _WeightCreator.maybe_materialize(weight)

            if "gate_up_proj_blocks" in name:
                new_name = name.replace("gate_up_proj_blocks", "w13_weight")
                weight = weight.view(
                    moe_num_global_experts, 2 * intermediate_size, -1
                ).contiguous()
                narrow_weight = weight[
                    moe_ep_rank_start:moe_ep_rank_end,
                    2 * moe_tp_rank_start : 2 * moe_tp_rank_end,
                    ...,
                ]
                _copy_into_param(params_dict[new_name], narrow_weight)
                loaded_params.add(new_name)

            elif "down_proj_blocks" in name:
                new_name = name.replace("down_proj_blocks", "w2_weight")
                weight = weight.view(
                    moe_num_global_experts, -1, intermediate_size // 2
                ).contiguous()
                narrow_weight = weight[
                    moe_ep_rank_start:moe_ep_rank_end,
                    ...,
                    moe_tp_rank_start // 2 : moe_tp_rank_end // 2,
                ]
                _copy_into_param(params_dict[new_name], narrow_weight)
                loaded_params.add(new_name)

            elif "gate_up_proj_scales" in name:
                new_name = name.replace("gate_up_proj_scales", "w13_weight_scale")
                narrow_weight = weight[
                    moe_ep_rank_start:moe_ep_rank_end,
                    2 * moe_tp_rank_start : 2 * moe_tp_rank_end,
                    ...,
                ]
                _copy_into_param(params_dict[new_name], narrow_weight)
                loaded_params.add(new_name)

            elif "down_proj_scales" in name:
                new_name = name.replace("down_proj_scales", "w2_weight_scale")
                narrow_weight = weight[
                    moe_ep_rank_start:moe_ep_rank_end,
                    ...,
                    moe_tp_rank_start // mxfp4_block : moe_tp_rank_end // mxfp4_block,
                ]
                _copy_into_param(params_dict[new_name], narrow_weight)
                loaded_params.add(new_name)

            elif "gate_up_proj_bias" in name:
                new_name = name.replace("gate_up_proj_bias", "w13_weight_bias")
                narrow_weight = weight[
                    moe_ep_rank_start:moe_ep_rank_end,
                    2 * moe_tp_rank_start : 2 * moe_tp_rank_end,
                ]
                _copy_into_param(params_dict[new_name], narrow_weight)
                loaded_params.add(new_name)

            elif "down_proj_bias" in name:
                new_name = name.replace("down_proj_bias", "w2_weight_bias")
                narrow_weight = weight[moe_ep_rank_start:moe_ep_rank_end, ...]
                if moe_tp_rank != 0:
                    narrow_weight = torch.zeros_like(narrow_weight)
                _copy_into_param(params_dict[new_name], narrow_weight)
                loaded_params.add(new_name)

        return loaded_params

    def _load_mxfp4_per_expert_weights(
        self,
        weights,
        *,
        params_dict,
        moe_tp_rank_start: int,
        moe_tp_rank_end: int,
        moe_ep_rank_start: int,
        moe_ep_rank_end: int,
        moe_tp_rank: int,
        copy_into_param,
        mxfp4_block: int,
    ):
        """Load the AMD-Quark per-expert MXFP4 + FP8 input-scale checkpoint.

        Tensor names look like
        ``model.layers.{l}.mlp.experts.{e}.{gate_up_proj,down_proj}.{weight,
        weight_scale,bias,input_scale}`` and shapes match the existing fused
        ``w13_*`` / ``w2_*`` parameters once the per-expert tensors are
        stacked along the expert dimension.
        """
        loaded_params: set = set()

        per_expert_re = re.compile(
            r"^(?P<base>.*\.experts\.)(?P<expert>\d+)\.(?P<proj>gate_up_proj|down_proj)\.(?P<kind>weight_scale|weight|bias|input_scale)$"
        )

        for name, weight in weights:
            weight = _WeightCreator.maybe_materialize(weight)

            match = per_expert_re.match(name)
            if match is None:
                # ``router`` and other non-expert weights are emitted to the
                # generic loader by the caller; if we still hit one here it is
                # an unexpected name.
                continue

            base = match.group("base")
            expert_id = int(match.group("expert"))
            proj = match.group("proj")
            kind = match.group("kind")

            if not (moe_ep_rank_start <= expert_id < moe_ep_rank_end):
                continue
            local_expert_id = expert_id - moe_ep_rank_start

            if proj == "gate_up_proj":
                if kind == "weight":
                    target = base + "w13_weight"
                elif kind == "weight_scale":
                    target = base + "w13_weight_scale"
                elif kind == "bias":
                    target = base + "w13_weight_bias"
                else:  # input_scale
                    target = base + "w13_input_scale"
            else:  # down_proj
                if kind == "weight":
                    target = base + "w2_weight"
                elif kind == "weight_scale":
                    target = base + "w2_weight_scale"
                elif kind == "bias":
                    target = base + "w2_weight_bias"
                else:  # input_scale
                    target = base + "w2_input_scale"

            if target not in params_dict:
                # The active backend (e.g. plain MXFP4 without FP8 act) may
                # not allocate ``input_scale`` parameters; just skip.
                if kind == "input_scale":
                    continue
                raise KeyError(f"missing target parameter {target!r} for {name!r}")
            param = params_dict[target]

            if kind == "input_scale":
                # Per-tensor static FP8 activation scale; broadcast scalar
                # into the per-expert slot.
                param.data[local_expert_id] = (
                    weight.detach().to(torch.float32).reshape(())
                )
                loaded_params.add(target)
                continue

            if proj == "gate_up_proj":
                # Per-expert tensor shapes:
                #   weight:        (2*intermediate, hidden//2) uint8
                #   weight_scale:  (2*intermediate, hidden//mxfp4_block) uint8
                #   bias:          (2*intermediate,) bf16
                # The fused parameter slot is sharded along the (output)
                # intermediate dimension.
                if kind == "bias":
                    narrow = weight[2 * moe_tp_rank_start : 2 * moe_tp_rank_end]
                else:
                    narrow = weight[2 * moe_tp_rank_start : 2 * moe_tp_rank_end, :]
                copy_into_param(param.data[local_expert_id], narrow)
            else:  # down_proj
                # Per-expert tensor shapes:
                #   weight:        (hidden, intermediate//2) uint8
                #   weight_scale:  (hidden, intermediate//mxfp4_block) uint8
                #   bias:          (hidden,) bf16
                # Down_proj is sharded along the (input) intermediate
                # dimension.
                if kind == "bias":
                    if moe_tp_rank != 0:
                        narrow = torch.zeros_like(weight)
                    else:
                        narrow = weight
                elif kind == "weight":
                    narrow = weight[:, moe_tp_rank_start // 2 : moe_tp_rank_end // 2]
                else:  # weight_scale
                    narrow = weight[
                        :,
                        moe_tp_rank_start
                        // mxfp4_block : moe_tp_rank_end
                        // mxfp4_block,
                    ]
                copy_into_param(param.data[local_expert_id], narrow)

            loaded_params.add(target)

        return loaded_params


EntryClass = GptOssForCausalLM
