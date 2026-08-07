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

"""Configuration for the GLM-5.3-Flash multimodal model."""

from __future__ import annotations

from typing import Any

from tokenspeed.runtime.configs.base_config import BaseConfig, TextConfigBase
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)

_ATTENTION_LAYER = "attention"
_LINEAR_ATTENTION_LAYER = LINEAR_ATTENTION


class Glm53FlashVisionConfig(BaseConfig):
    """GLM-5.3-Flash vision tower configuration."""

    model_type = "glm53_flash_vision"

    depth: int = 24
    hidden_size: int = 1024
    hidden_act: str = "silu"
    attention_bias: bool = True
    attention_dropout: float = 0.0
    num_heads: int = 16
    in_channels: int = 3
    image_size: int = 448
    patch_size: int = 14
    rms_norm_eps: float = 1e-5
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 4096
    intermediate_size: int = 4096
    initializer_range: float = 0.02
    projection_intermediate_size: int = 10240
    swiglu_limit: float | None = 10.0


class Glm53FlashTextConfig(TextConfigBase):
    """Text configuration and mixed-layer protocol for GLM-5.3-Flash.

    ``layer_types`` retains the checkpoint's native KDA/DSA vocabulary.
    ``layers_block_type`` and ``paged_cache_layer_types`` expose the shared
    TokenSpeed hybrid-layer and cache vocabularies used by Kimi-K3.
    """

    model_type = "glm53_flash_text"

    vocab_size: int = 154880
    hidden_size: int = 4096
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 45
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    n_shared_experts: int = 1
    n_routed_experts: int = 288
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_head_dim: int = 256
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    qk_nope_head_dim: int = 256
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int = 154820
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    head_dim: int = 0
    layer_types: list[str] | None = None
    indexer_types: list[str] | None = None
    mlp_layer_types: list[str] | None = None
    swiglu_limit: float = 10.0
    linear_attn_config: dict | None = None
    linear_head_dim: int = 128
    linear_num_heads: int = 64
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: float = -5.0
    gate_lower_bound: float | None = None
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    index_kpool: int = 4
    index_kpool_always_select_tail: bool = True
    index_kpool_compress: bool = True
    index_skip_topk_offset: int = 2
    index_topk_freq: int = 1
    indexer_rope_interleave: bool = True
    mhc: bool = True
    moe_router_dtype: str = "float32"
    num_nextn_predict_layers: int = 1
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    first_k_dense_replace: int = 3

    def __post_init__(self, **kwargs: Any) -> None:
        linear_attn_config = self.linear_attn_config
        linear_head_dim = self.linear_head_dim
        linear_num_heads = self.linear_num_heads
        linear_conv_kernel_dim = self.linear_conv_kernel_dim
        linear_lower_bound = self.linear_lower_bound
        gate_lower_bound = self.gate_lower_bound

        if linear_attn_config is not None:
            linear_head_dim = linear_attn_config.get("head_dim", linear_head_dim)
            linear_num_heads = linear_attn_config.get("num_heads", linear_num_heads)
            linear_conv_kernel_dim = linear_attn_config.get(
                "short_conv_kernel_size", linear_conv_kernel_dim
            )
            gate_lower_bound = linear_attn_config.get(
                "gate_lower_bound", gate_lower_bound
            )
        if gate_lower_bound is not None:
            linear_lower_bound = gate_lower_bound

        layer_types = self.layer_types
        if layer_types is None and linear_attn_config is not None:
            kda_layers = set(linear_attn_config.get("kda_layers", ()))
            full_attn_layers = set(linear_attn_config.get("full_attn_layers", ()))
            layer_types = [
                (
                    _LINEAR_ATTENTION_LAYER
                    if layer_id in kda_layers
                    else "deepseek_sparse_attention"
                )
                for layer_id in range(self.num_hidden_layers)
            ]
            unknown_layers = set(range(self.num_hidden_layers)) - (
                kda_layers | full_attn_layers
            )
            if unknown_layers:
                raise ValueError(
                    "linear_attn_config must classify every layer as KDA or full attention"
                )
        self.layer_types = (
            layer_types
            if layer_types is not None
            else [
                (
                    "deepseek_sparse_attention"
                    if layer_id % 4 == 3
                    else _LINEAR_ATTENTION_LAYER
                )
                for layer_id in range(self.num_hidden_layers)
            ]
        )

        indexer_types = self.indexer_types
        self.indexer_types = (
            indexer_types
            if indexer_types is not None
            else [
                (
                    "full"
                    if max(layer_id - self.index_skip_topk_offset + 1, 0)
                    % max(self.index_topk_freq, 1)
                    == 0
                    else "shared"
                )
                for layer_id in range(self.num_hidden_layers)
            ]
        )

        mlp_layer_types = self.mlp_layer_types
        dense_layers = min(self.first_k_dense_replace, self.num_hidden_layers)
        self.mlp_layer_types = (
            mlp_layer_types
            if mlp_layer_types is not None
            else ["dense"] * dense_layers
            + ["sparse"] * (self.num_hidden_layers - dense_layers)
        )

        self.linear_head_dim = linear_head_dim
        self.linear_num_heads = linear_num_heads
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_lower_bound = linear_lower_bound
        self.gate_lower_bound = linear_lower_bound
        self.linear_attn_config = {
            "num_heads": linear_num_heads,
            "head_dim": linear_head_dim,
            "short_conv_kernel_size": linear_conv_kernel_dim,
            "gate_lower_bound": linear_lower_bound,
            "kda_layers": self.linear_layer_ids,
            "full_attn_layers": self.full_attention_layer_ids,
        }

        super().__post_init__(**kwargs)

    def is_kda_layer(self, layer_id: int) -> bool:
        return self.layer_types[layer_id] == _LINEAR_ATTENTION_LAYER

    def is_dsa_layer(self, layer_id: int) -> bool:
        return self.layer_types[layer_id] == "deepseek_sparse_attention"

    @property
    def layers_block_type(self) -> list[str]:
        """Per-layer KDA/attention labels used by hybrid model code."""
        return [
            (
                _LINEAR_ATTENTION_LAYER
                if layer_type == _LINEAR_ATTENTION_LAYER
                else _ATTENTION_LAYER
            )
            for layer_type in self.layer_types
        ]

    @property
    def paged_cache_layer_types(self) -> list[str]:
        """Translate compute-layer labels to TokenSpeed paged-cache labels."""
        return [
            FULL_ATTENTION if layer_type == _ATTENTION_LAYER else layer_type
            for layer_type in self.layers_block_type
        ]

    @property
    def linear_layer_ids(self) -> list[int]:
        return [
            layer_id
            for layer_id, layer_type in enumerate(self.layers_block_type)
            if layer_type == _LINEAR_ATTENTION_LAYER
        ]

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return [
            layer_id
            for layer_id, layer_type in enumerate(self.layers_block_type)
            if layer_type == _ATTENTION_LAYER
        ]


class Glm53FlashConfig(BaseConfig):
    """Top-level multimodal GLM-5.3-Flash configuration."""

    model_type = "glm53_flash"
    sub_configs = {
        "vision_config": Glm53FlashVisionConfig,
        "text_config": Glm53FlashTextConfig,
    }

    text_config: Glm53FlashTextConfig | dict | None = None
    vision_config: Glm53FlashVisionConfig | dict | None = None
    image_token_id: int = 154854
    video_token_id: int = 154855
    image_start_token_id: int = 154830
    image_end_token_id: int = 154831
    video_start_token_id: int = 154832
    video_end_token_id: int = 154833
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs: Any) -> None:
        if self.text_config is None:
            self.text_config = Glm53FlashTextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = Glm53FlashTextConfig(**self.text_config)

        text_swiglu_limit = getattr(self.text_config, "swiglu_limit", None)
        if self.vision_config is None:
            self.vision_config = Glm53FlashVisionConfig(swiglu_limit=text_swiglu_limit)
        elif isinstance(self.vision_config, dict):
            vision_values = dict(self.vision_config)
            vision_values.setdefault("swiglu_limit", text_swiglu_limit)
            self.vision_config = Glm53FlashVisionConfig(**vision_values)
        if self.vision_config.swiglu_limit is None:
            raise ValueError("GLM-5.3-Flash vision_config requires swiglu_limit")

        self.vision_config.out_hidden_size = self.text_config.hidden_size

        super().__post_init__(**kwargs)

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size


__all__ = [
    "Glm53FlashConfig",
    "Glm53FlashTextConfig",
    "Glm53FlashVisionConfig",
]
