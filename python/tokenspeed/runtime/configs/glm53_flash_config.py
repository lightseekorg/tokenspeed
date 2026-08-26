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

import torch
from transformers.configuration_utils import PretrainedConfig

from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)

_ATTENTION_LAYER = "attention"
_LINEAR_ATTENTION_LAYER = LINEAR_ATTENTION
GLM53_FLASH_MODEL_TYPE = "glm53_flash"
GLM53_FLASH_TEXT_MODEL_TYPE = "glm53_flash_text"
GLM53_FLASH_VISION_MODEL_TYPE = "glm53_flash_vision"
LEGACY_GLM53_FLASH_MODEL_TYPE = "glm5_next"


class Glm53FlashVisionConfig(PretrainedConfig):
    """GLM-5.3-Flash vision tower configuration."""

    model_type = GLM53_FLASH_VISION_MODEL_TYPE

    def __init__(
        self,
        depth: int = 24,
        hidden_size: int = 1024,
        hidden_act: str = "silu",
        attention_bias: bool = True,
        attention_dropout: float = 0.0,
        num_heads: int = 16,
        in_channels: int = 3,
        image_size: int = 448,
        patch_size: int = 14,
        rms_norm_eps: float = 1e-5,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 4096,
        intermediate_size: int = 4096,
        initializer_range: float = 0.02,
        projection_intermediate_size: int = 10240,
        swiglu_limit: float | None = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.rms_norm_eps = rms_norm_eps
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.projection_intermediate_size = projection_intermediate_size
        self.swiglu_limit = swiglu_limit


class Glm53FlashTextConfig(PretrainedConfig):
    """Text configuration and mixed-layer protocol for GLM-5.3-Flash.

    ``layer_types`` retains the checkpoint's native KDA/DSA vocabulary.
    ``layers_block_type`` and ``paged_cache_layer_types`` expose the shared
    TokenSpeed hybrid-layer and cache vocabularies used by Kimi-K3.
    """

    model_type = GLM53_FLASH_TEXT_MODEL_TYPE

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 45,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 64,
        n_shared_experts: int = 1,
        n_routed_experts: int = 288,
        routed_scaling_factor: float = 2.5,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 1536,
        qk_head_dim: int = 256,
        qk_rope_head_dim: int = 0,
        v_head_dim: int = 256,
        qk_nope_head_dim: int = 256,
        n_group: int = 1,
        topk_group: int = 1,
        num_experts_per_tok: int = 8,
        norm_topk_prob: bool = True,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int = 154820,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        index_topk: int = 2048,
        index_head_dim: int = 128,
        index_n_heads: int = 32,
        head_dim: int = 0,
        layer_types: list[str] | None = None,
        indexer_types: list[str] | None = None,
        mlp_layer_types: list[str] | None = None,
        swiglu_limit: float = 10.0,
        linear_attn_config: dict | None = None,
        linear_head_dim: int = 128,
        linear_num_heads: int = 64,
        linear_conv_kernel_dim: int = 4,
        linear_lower_bound: float = -5.0,
        gate_lower_bound: float | None = None,
        hc_mult: int = 4,
        hc_eps: float = 1e-6,
        hc_sinkhorn_iters: int = 20,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        index_kpool: int = 4,
        index_kpool_always_select_tail: bool = True,
        index_kpool_compress: bool = True,
        index_skip_topk_offset: int = 2,
        index_topk_freq: int = 1,
        indexer_rope_interleave: bool = True,
        mhc: bool = True,
        moe_router_dtype: str = "float32",
        num_nextn_predict_layers: int = 1,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        first_k_dense_replace: int = 3,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_head_dim = qk_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.head_dim = head_dim
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

        if layer_types is None and linear_attn_config is not None:
            kda_layers = set(linear_attn_config.get("kda_layers", ()))
            full_attn_layers = set(linear_attn_config.get("full_attn_layers", ()))
            layer_types = [
                (
                    _LINEAR_ATTENTION_LAYER
                    if layer_id in kda_layers
                    else "deepseek_sparse_attention"
                )
                for layer_id in range(num_hidden_layers)
            ]
            unknown_layers = set(range(num_hidden_layers)) - (
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
                for layer_id in range(num_hidden_layers)
            ]
        )
        self.indexer_types = (
            indexer_types
            if indexer_types is not None
            else [
                (
                    "full"
                    if max(layer_id - index_skip_topk_offset + 1, 0)
                    % max(index_topk_freq, 1)
                    == 0
                    else "shared"
                )
                for layer_id in range(num_hidden_layers)
            ]
        )
        dense_layers = min(first_k_dense_replace, num_hidden_layers)
        self.mlp_layer_types = (
            mlp_layer_types
            if mlp_layer_types is not None
            else ["dense"] * dense_layers
            + ["sparse"] * (num_hidden_layers - dense_layers)
        )
        self.swiglu_limit = swiglu_limit
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
        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.index_kpool = index_kpool
        self.index_kpool_always_select_tail = index_kpool_always_select_tail
        self.index_kpool_compress = index_kpool_compress
        self.index_skip_topk_offset = index_skip_topk_offset
        self.index_topk_freq = index_topk_freq
        self.indexer_rope_interleave = indexer_rope_interleave
        self.mhc = mhc
        self.moe_router_dtype = moe_router_dtype
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.first_k_dense_replace = first_k_dense_replace
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def mamba2_cache_params(self):
        """KDA per-request state spec consumed by the hybrid KV-cache allocator.

        Returns ``(conv_state_shape, temporal_state_shape, conv_dtype,
        ssm_dtype, mamba_layer_ids)`` — the same contract as Kimi-K3's
        KDA: three short causal convolutions (q/k/v) per layer and a
        per-head ``head_dim x head_dim`` delta-rule recurrent state in fp32.
        """
        from tokenspeed.runtime.distributed.utils import divide
        from tokenspeed.runtime.utils.env import global_server_args_dict

        attn_tp_size = global_server_args_dict["mapping"].attn.tp_size
        conv_state_shape = (
            divide(
                3 * self.linear_num_heads * self.linear_head_dim,
                attn_tp_size,
            ),
            self.linear_conv_kernel_dim - 1,
        )
        temporal_state_shape = (
            divide(self.linear_num_heads, attn_tp_size),
            self.linear_head_dim,
            self.linear_head_dim,
        )
        return (
            conv_state_shape,
            temporal_state_shape,
            torch.bfloat16,
            torch.float32,
            self.linear_layer_ids,
        )

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


class Glm53FlashConfig(PretrainedConfig):
    """Top-level multimodal GLM-5.3-Flash configuration."""

    model_type = GLM53_FLASH_MODEL_TYPE
    text_config_cls = Glm53FlashTextConfig
    vision_config_cls = Glm53FlashVisionConfig

    def __init__(
        self,
        text_config: dict | Glm53FlashTextConfig | None = None,
        vision_config: dict | Glm53FlashVisionConfig | None = None,
        image_token_id: int = 154854,
        video_token_id: int = 154855,
        image_start_token_id: int = 154830,
        image_end_token_id: int = 154831,
        video_start_token_id: int = 154832,
        video_end_token_id: int = 154833,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        if text_config is None:
            text_config = self.text_config_cls()
        elif isinstance(text_config, dict):
            text_config = self.text_config_cls(**text_config)

        text_swiglu_limit = getattr(text_config, "swiglu_limit", None)
        if vision_config is None:
            vision_config = self.vision_config_cls(swiglu_limit=text_swiglu_limit)
        elif isinstance(vision_config, dict):
            vision_values = dict(vision_config)
            vision_values.setdefault("model_type", self.vision_config_cls.model_type)
            vision_values.setdefault("swiglu_limit", text_swiglu_limit)
            vision_config = self.vision_config_cls(**vision_values)
        if vision_config.swiglu_limit is None:
            raise ValueError("GLM-5.3-Flash vision_config requires swiglu_limit")

        self.text_config = text_config
        self.vision_config = vision_config
        self.vision_config.out_hidden_size = self.text_config.hidden_size
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size
