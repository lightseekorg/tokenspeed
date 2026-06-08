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

"""Qwen3.5: HF config → EngineModelSpec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from tokenspeed.runtime.configs.engine_spec import (
    ENGINE_SPEC_SCHEMA_VERSION,
    EngineModelSpec,
    GatedDeltaNetSpec,
    GQAAttentionSpec,
    MoEMLPSpec,
    Qwen35ModelSpec,
    RMSNormSpec,
    RopePositionSpec,
)
from tokenspeed.runtime.configs.qwen3_5_text_base_config import HybridLayerType
from tokenspeed.runtime.distributed.utils import divide
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.hf_transformers_utils import get_context_length


def _resolve_text_config(hf_config: Any) -> Any:
    text_cfg = getattr(hf_config, "text_config", None)
    if text_cfg is not None:
        return text_cfg
    model_type = getattr(hf_config, "model_type", None)
    if model_type in {"qwen3_5_text", "qwen3_5_moe_text"}:
        return hf_config
    raise ValueError(
        f"qwen3_5 adapter expected text_config or text-only config, got "
        f"model_type={model_type!r}"
    )


def _normalize_rope_scaling(text_cfg: Any) -> dict[str, Any] | None:
    rope_scaling = getattr(text_cfg, "rope_scaling", None)
    if rope_scaling is None:
        rope_parameters = getattr(text_cfg, "rope_parameters", None)
        if isinstance(rope_parameters, dict):
            rope_scaling = rope_parameters
    if rope_scaling is not None and not isinstance(rope_scaling, dict):
        rope_scaling = None
    return rope_scaling


@dataclass
class Qwen35RuntimeView:
    """Temporary bridge for ``models/qwen3_5.py`` until it reads ``body``."""

    model_type: str
    architectures: list[str]
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_act: str
    max_position_embeddings: int
    rms_norm_eps: float
    tie_word_embeddings: bool
    rope_theta: float
    rope_scaling: dict[str, Any] | None
    rope_parameters: dict[str, Any] | None
    attention_bias: bool
    partial_rotary_factor: float
    attn_output_gate: bool
    full_attention_interval: int
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    decoder_sparse_step: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    num_experts_per_tok: int
    num_experts: int
    norm_topk_prob: bool
    mlp_only_layers: list[int]
    torch_dtype: Any = None
    mapping: Any = None

    @classmethod
    def from_engine_spec(cls, spec: EngineModelSpec) -> Qwen35RuntimeView:
        body = spec.body
        if body.type != "qwen3_5":
            raise ValueError(f"expected qwen3_5 body, got {body.type!r}")
        attn = body.attention
        mlp = body.mlp
        linear = body.linear_attn
        rope_scaling = body.position.scaling
        return cls(
            model_type=body.text_model_type,
            architectures=list(spec.architectures),
            vocab_size=body.vocab_size,
            hidden_size=body.hidden_size,
            intermediate_size=body.dense_intermediate_size,
            num_hidden_layers=body.num_layers,
            num_attention_heads=attn.num_query_heads,
            num_key_value_heads=attn.num_kv_heads,
            head_dim=attn.head_dim,
            hidden_act=body.hidden_act,
            max_position_embeddings=body.position.max_position_embeddings,
            rms_norm_eps=body.norm.eps,
            tie_word_embeddings=body.tie_word_embeddings,
            rope_theta=body.position.rope_theta,
            rope_scaling=rope_scaling,
            rope_parameters=rope_scaling,
            attention_bias=attn.attention_bias,
            partial_rotary_factor=body.partial_rotary_factor,
            attn_output_gate=body.attn_output_gate,
            full_attention_interval=body.full_attention_interval,
            linear_conv_kernel_dim=linear.conv_kernel_dim,
            linear_key_head_dim=linear.key_head_dim,
            linear_value_head_dim=linear.value_head_dim,
            linear_num_key_heads=linear.num_key_heads,
            linear_num_value_heads=linear.num_value_heads,
            decoder_sparse_step=body.decoder_sparse_step,
            moe_intermediate_size=body.moe_intermediate_size,
            shared_expert_intermediate_size=body.shared_expert_intermediate_size,
            num_experts_per_tok=mlp.top_k,
            num_experts=mlp.num_experts,
            norm_topk_prob=mlp.norm_topk_prob,
            mlp_only_layers=list(body.mlp_only_layers),
        )

    @property
    def layers_block_type(self) -> list[str]:
        layer_type_list = []
        for layer_index in range(self.num_hidden_layers):
            if (layer_index + 1) % self.full_attention_interval == 0:
                layer_type_list.append(HybridLayerType.full_attention.value)
            else:
                layer_type_list.append(HybridLayerType.linear_attention.value)
        return layer_type_list

    @property
    def linear_layer_ids(self) -> list[int]:
        return [
            i
            for i, type_value in enumerate(self.layers_block_type)
            if type_value == HybridLayerType.linear_attention.value
        ]

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return [
            i
            for i, type_value in enumerate(self.layers_block_type)
            if type_value == HybridLayerType.full_attention.value
        ]

    @property
    def mamba2_cache_params(self):
        from tokenspeed.runtime.utils.env import global_server_args_dict

        self.mapping = global_server_args_dict["mapping"]
        attn_tp_size = self.mapping.attn.tp_size

        conv_dim = (
            self.linear_key_head_dim * self.linear_num_key_heads * 2
            + self.linear_value_head_dim * self.linear_num_value_heads
        )
        conv_state_shape = (
            divide(conv_dim, attn_tp_size),
            self.linear_conv_kernel_dim - 1,
        )
        temporal_state_shape = (
            divide(self.linear_num_value_heads, attn_tp_size),
            self.linear_key_head_dim,
            self.linear_value_head_dim,
        )
        conv_dtype = torch.bfloat16
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        ssm_dtype = dtype_map[envs.TOKENSPEED_MAMBA_SSM_DTYPE.get()]
        mamba_layers = self.linear_layer_ids
        return (
            conv_state_shape,
            temporal_state_shape,
            conv_dtype,
            ssm_dtype,
            mamba_layers,
        )

    @property
    def mamba_cache_per_req(self) -> int:
        conv_state_shape, temporal_state_shape, conv_dtype, ssm_dtype, mamba_layers = (
            self.mamba2_cache_params
        )
        mamba_layers_len = len(mamba_layers)
        return (
            int(np.prod(conv_state_shape)) * conv_dtype.itemsize
            + int(np.prod(temporal_state_shape)) * ssm_dtype.itemsize
        ) * mamba_layers_len


def from_hugging_face(hf_config: Any) -> EngineModelSpec:
    shell_model_type = getattr(hf_config, "model_type", None)
    if shell_model_type not in {"qwen3_5", "qwen3_5_moe"}:
        raise ValueError(
            f"expected model_type in ('qwen3_5', 'qwen3_5_moe'), got "
            f"{shell_model_type!r}"
        )

    text_cfg = _resolve_text_config(hf_config)
    text_model_type = str(getattr(text_cfg, "model_type", f"{shell_model_type}_text"))

    architectures = [
        str(name)
        for name in (getattr(hf_config, "architectures", None) or [])
        if isinstance(name, str) and name
    ]
    if not architectures:
        raise ValueError("qwen3_5 config must declare architectures")

    num_query_heads = int(text_cfg.num_attention_heads)
    num_kv_heads = int(getattr(text_cfg, "num_key_value_heads", num_query_heads))
    head_dim = int(
        getattr(text_cfg, "head_dim", None) or text_cfg.hidden_size // num_query_heads
    )
    rope_scaling = _normalize_rope_scaling(text_cfg)
    full_attention_interval = getattr(text_cfg, "full_attention_interval", None)
    if full_attention_interval is None:
        raise ValueError("qwen3_5 text config must declare full_attention_interval")

    dtype = getattr(hf_config, "dtype", None)
    if dtype is None:
        torch_dtype = getattr(hf_config, "torch_dtype", None)
        dtype = str(torch_dtype).removeprefix("torch.") if torch_dtype else None
    else:
        dtype = str(dtype)

    quant_cfg = getattr(hf_config, "quantization_config", None)
    if quant_cfg is None:
        quant_cfg = getattr(hf_config, "compression_config", None)

    mlp_only_layers = tuple(
        int(layer_id) for layer_id in (getattr(text_cfg, "mlp_only_layers", None) or [])
    )

    body = Qwen35ModelSpec(
        type="qwen3_5",
        architecture_kind="decoder_only_lm",
        text_model_type=text_model_type,
        num_layers=int(text_cfg.num_hidden_layers),
        hidden_size=int(text_cfg.hidden_size),
        vocab_size=int(text_cfg.vocab_size),
        hidden_act=str(getattr(text_cfg, "hidden_act", "silu")),
        tie_word_embeddings=bool(getattr(text_cfg, "tie_word_embeddings", False)),
        attention=GQAAttentionSpec(
            type="gqa",
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_dim=head_dim,
            attention_bias=bool(getattr(text_cfg, "attention_bias", False)),
            use_qk_norm=False,
            qk_norm_type=None,
        ),
        mlp=MoEMLPSpec(
            type="moe",
            num_experts=int(getattr(text_cfg, "num_experts", 0) or 0),
            top_k=int(getattr(text_cfg, "num_experts_per_tok", 0) or 0),
            intermediate_size=int(
                getattr(text_cfg, "moe_intermediate_size", None)
                or getattr(text_cfg, "intermediate_size", 0)
            ),
            scoring_func="",
            use_routing_bias=False,
            norm_topk_prob=bool(getattr(text_cfg, "norm_topk_prob", True)),
        ),
        linear_attn=GatedDeltaNetSpec(
            type="gated_delta_net",
            conv_kernel_dim=int(text_cfg.linear_conv_kernel_dim),
            key_head_dim=int(text_cfg.linear_key_head_dim),
            value_head_dim=int(text_cfg.linear_value_head_dim),
            num_key_heads=int(text_cfg.linear_num_key_heads),
            num_value_heads=int(text_cfg.linear_num_value_heads),
        ),
        norm=RMSNormSpec(
            type="rmsnorm",
            eps=float(getattr(text_cfg, "rms_norm_eps", 1e-6)),
        ),
        position=RopePositionSpec(
            type="rope",
            rope_theta=float(text_cfg.rope_theta),
            scaling=rope_scaling,
            max_position_embeddings=int(text_cfg.max_position_embeddings),
            context_len=int(get_context_length(text_cfg)),
        ),
        full_attention_interval=int(full_attention_interval),
        partial_rotary_factor=float(getattr(text_cfg, "partial_rotary_factor", 0.25)),
        attn_output_gate=bool(getattr(text_cfg, "attn_output_gate", True)),
        decoder_sparse_step=int(getattr(text_cfg, "decoder_sparse_step", 1)),
        mlp_only_layers=mlp_only_layers,
        dense_intermediate_size=int(text_cfg.intermediate_size),
        moe_intermediate_size=int(
            getattr(text_cfg, "moe_intermediate_size", text_cfg.intermediate_size)
        ),
        shared_expert_intermediate_size=int(
            getattr(
                text_cfg,
                "shared_expert_intermediate_size",
                getattr(text_cfg, "moe_intermediate_size", text_cfg.intermediate_size),
            )
        ),
    )

    return EngineModelSpec(
        schema_version=ENGINE_SPEC_SCHEMA_VERSION,
        model_type=shell_model_type,
        architecture=architectures[0],
        architectures=tuple(architectures),
        dtype=dtype,
        quantization=quant_cfg,
        body=body,
    )


__all__ = ["Qwen35RuntimeView", "from_hugging_face"]
