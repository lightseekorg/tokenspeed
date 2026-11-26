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

"""Configuration classes for the Hy4-flash multimodal MoE model and DFly2 drafter."""

from __future__ import annotations

import copy
from typing import Any

from transformers.configuration_utils import PretrainedConfig

from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
)

DEEPSEEK_SPARSE_ATTENTION = "deepseek_sparse_attention"
# Canonical indexed-DSA layer name matches HF transformers' ALLOWED_LAYER_TYPES
# (`deepseek_sparse_attention`), while still accepting legacy `"indexed_attention"`.
INDEXED_ATTENTION = DEEPSEEK_SPARSE_ATTENTION
SLIDING_ATTENTION = "sliding_attention"

_INDEXED_LAYER_NAMES = frozenset({"indexed_attention", DEEPSEEK_SPARSE_ATTENTION})
_SLIDING_LAYER_NAMES = frozenset({SLIDING_ATTENTION})

_DEFAULT_MROPE_SECTION = [12, 10, 10]
_DEFAULT_INDEXED_ROPE_THETA = 10000000.0
_DEFAULT_SLIDING_ROPE_THETA = 10000.0


def _normalize_layer_type(layer_type: str) -> str:
    if layer_type in _INDEXED_LAYER_NAMES:
        return INDEXED_ATTENTION
    if layer_type in _SLIDING_LAYER_NAMES:
        return SLIDING_ATTENTION
    raise ValueError(
        f"Unsupported HYV4Flash layer_type={layer_type!r}; expected one of "
        f"{sorted(_INDEXED_LAYER_NAMES | _SLIDING_LAYER_NAMES)}"
    )


def _default_layer_types(num_hidden_layers: int) -> list[str]:
    return [
        INDEXED_ATTENTION if i < 2 or (i - 1) % 4 == 0 else SLIDING_ATTENTION
        for i in range(num_hidden_layers)
    ]


def _normalize_rope_parameters(
    raw_rope_parameters: dict[str, Any] | None,
    *,
    rope_theta: float | None = None,
    rope_scaling: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {
        INDEXED_ATTENTION: {
            "rope_type": "default",
            "rope_theta": (
                float(rope_theta)
                if rope_theta is not None
                else _DEFAULT_INDEXED_ROPE_THETA
            ),
            "mrope_section": list(_DEFAULT_MROPE_SECTION),
            "mrope_interleaved": True,
        },
        SLIDING_ATTENTION: {
            "rope_type": "default",
            "rope_theta": (
                float(rope_theta)
                if rope_theta is not None
                else _DEFAULT_SLIDING_ROPE_THETA
            ),
            "mrope_section": list(_DEFAULT_MROPE_SECTION),
            "mrope_interleaved": True,
        },
    }
    if isinstance(raw_rope_parameters, dict):
        if any(
            key in raw_rope_parameters
            for key in (
                INDEXED_ATTENTION,
                DEEPSEEK_SPARSE_ATTENTION,
                SLIDING_ATTENTION,
            )
        ):
            for key, value in raw_rope_parameters.items():
                if not isinstance(value, dict):
                    continue
                canonical = _normalize_layer_type(key)
                normalized[canonical].update(value)
        else:
            for canonical in (INDEXED_ATTENTION, SLIDING_ATTENTION):
                normalized[canonical].update(raw_rope_parameters)
    if isinstance(rope_scaling, dict):
        normalized[INDEXED_ATTENTION].update(rope_scaling)
    for canonical in (INDEXED_ATTENTION, SLIDING_ATTENTION):
        normalized[canonical].setdefault("rope_type", "default")
        normalized[canonical].setdefault("mrope_section", list(_DEFAULT_MROPE_SECTION))
        normalized[canonical].setdefault("mrope_interleaved", True)
        if "rope_theta" not in normalized[canonical]:
            normalized[canonical]["rope_theta"] = (
                _DEFAULT_INDEXED_ROPE_THETA
                if canonical == INDEXED_ATTENTION
                else _DEFAULT_SLIDING_ROPE_THETA
            )
    return normalized


class HYV4FlashVisionConfig(PretrainedConfig):
    """SigLIP2-based vision tower configuration for Hy4-flash."""

    model_type = "hy_v4_vl_vision"
    base_config_key = "vision_config"
    attribute_map = {
        "attention_heads": "num_attention_heads",
        "layer_norm_eps": "rms_norm_eps",
    }

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = None,
        num_channels: int = 3,
        out_hidden_size: int = 3072,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        max_image_size: int = 2048,
        interpolate_mode: str = "bilinear",
        hidden_act: str = "gelu",
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        spatial_patch_size: int | None = None,
        **kwargs,
    ) -> None:
        # Published checkpoints store vision_config.model_type = "hy_v4_flash";
        # avoid recursive AutoConfig resolution by pinning to hy_v4_vl_vision.
        del spatial_patch_size
        kwargs.pop("model_type", None)
        super().__init__(**kwargs)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = (
            int(num_key_value_heads)
            if num_key_value_heads is not None
            else self.num_attention_heads
        )
        self.num_channels = int(num_channels)
        self.out_hidden_size = int(out_hidden_size)
        self.patch_size = int(patch_size)
        self.spatial_merge_size = int(spatial_merge_size)
        self.max_image_size = int(max_image_size)
        self.interpolate_mode = str(interpolate_mode)
        self.hidden_act = str(hidden_act)
        self.rms_norm_eps = float(rms_norm_eps)
        self.attention_dropout = float(attention_dropout)


class HYV4FlashTextConfig(PretrainedConfig):
    """Text backbone configuration for Hy4-flash (Hybrid DSA/SWA MLA + LatentMoE + iHC + Engram)."""

    model_type = "hy_v4_flash_text"
    base_config_key = "text_config"
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    def __init__(
        self,
        vocab_size: int = 120832,
        hidden_size: int = 3072,
        intermediate_size: int = 13824,
        moe_intermediate_size: int = 1536,
        routed_expert_hidden_size: int = 1536,
        num_hidden_layers: int = 42,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        swa_num_attention_heads: int = 64,
        swa_qk_nope_head_dim: int | None = None,
        swa_qk_rope_head_dim: int | None = None,
        swa_v_head_dim: int | None = None,
        swa_qk_head_dim: int | None = None,
        sliding_window: int = 512,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        initializer_range: float = 0.006,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int | None = 120002,
        bos_token_id: int | None = 120000,
        eos_token_id: int | list[int] | None = 120025,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        n_routed_experts: int = 512,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 16,
        routed_scaling_factor: float = 3.998,
        norm_topk_prob: bool = True,
        n_group: int = 1,
        topk_group: int = 1,
        q_lora_rank: int = 1024,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        qk_head_dim: int | None = None,
        head_dim: int | None = None,
        mlp_layer_types: list[str] | None = None,
        layer_types: list[str] | None = None,
        index_topk: int = 2048,
        index_head_dim: int = 128,
        index_n_heads: int = 16,
        index_kpool: int = 4,
        compressed_indexer_compress_ratio: int | None = None,
        compressed_indexer_num_attention_heads: int | None = None,
        compressed_indexer_head_dim: int | None = None,
        compressed_indexer_group_topk: int | None = None,
        compressed_indexer_candidate_topk: int | None = None,
        compressed_indexer_skip_fine_select: bool = True,
        hc_mult: int = 4,
        hc_magnitude: float = 2.0,
        hc_eps: float = 1e-6,
        learnable_sink: bool = True,
        learnable_sink_init: float = 0.0,
        gated_mla: bool = True,
        attn_post_proj_norm: bool = True,
        enable_lm_head_fp32: bool = True,
        swiglu_limit: float = 10.0,
        rope_parameters: dict[str, Any] | None = None,
        rope_theta: float | None = None,
        rope_scaling: dict[str, Any] | None = None,
        engram_layer_ids: list[int] | None = None,
        engram_max_ngram_size: int = 3,
        engram_n_head_per_ngram: int = 8,
        engram_n_embed_per_ngram: int = 768,
        engram_layer_head_vocab_sizes: dict[str, list[list[int]]] | None = None,
        engram_pad_id: int = 120002,
        engram_fp8_embedding: bool = False,
        engram_seed: int = 0,
        engram_lookup_table_size: int = 120818,
        image_token_id: int = 120120,
        video_token_id: int = 120124,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        **kwargs,
    ) -> None:
        if compressed_indexer_compress_ratio is not None:
            index_kpool = int(compressed_indexer_compress_ratio)
        if compressed_indexer_num_attention_heads is not None:
            index_n_heads = int(compressed_indexer_num_attention_heads)
        if compressed_indexer_head_dim is not None:
            index_head_dim = int(compressed_indexer_head_dim)
        if compressed_indexer_candidate_topk is not None:
            index_topk = int(compressed_indexer_candidate_topk)
        elif compressed_indexer_group_topk is not None:
            index_topk = int(compressed_indexer_group_topk) * int(index_kpool)

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.routed_expert_hidden_size = int(routed_expert_hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = (
            int(num_key_value_heads)
            if num_key_value_heads is not None
            else self.num_attention_heads
        )
        self.swa_num_attention_heads = int(swa_num_attention_heads)
        self.sliding_window = int(sliding_window)
        self.hidden_act = str(hidden_act)
        self.max_position_embeddings = int(max_position_embeddings)
        self.initializer_range = float(initializer_range)
        self.rms_norm_eps = float(rms_norm_eps)
        self.use_cache = bool(use_cache)
        self.attention_bias = bool(attention_bias)
        self.attention_dropout = float(attention_dropout)
        self.n_routed_experts = int(n_routed_experts)
        self.n_shared_experts = int(n_shared_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.n_group = int(n_group)
        self.topk_group = int(topk_group)
        self.q_lora_rank = int(q_lora_rank)
        self.kv_lora_rank = int(kv_lora_rank)
        self.qk_nope_head_dim = int(qk_nope_head_dim)
        self.qk_rope_head_dim = int(qk_rope_head_dim)
        self.v_head_dim = int(v_head_dim)
        self.qk_head_dim = (
            int(qk_head_dim)
            if qk_head_dim is not None
            else self.qk_nope_head_dim + self.qk_rope_head_dim
        )
        if self.qk_head_dim != self.qk_nope_head_dim + self.qk_rope_head_dim:
            raise ValueError(
                f"qk_head_dim ({self.qk_head_dim}) must equal "
                f"qk_nope_head_dim ({self.qk_nope_head_dim}) + "
                f"qk_rope_head_dim ({self.qk_rope_head_dim})"
            )
        self.head_dim = int(head_dim) if head_dim is not None else self.qk_head_dim

        self.swa_qk_nope_head_dim = (
            int(swa_qk_nope_head_dim)
            if swa_qk_nope_head_dim is not None
            else self.qk_nope_head_dim
        )
        self.swa_qk_rope_head_dim = (
            int(swa_qk_rope_head_dim)
            if swa_qk_rope_head_dim is not None
            else self.qk_rope_head_dim
        )
        self.swa_v_head_dim = (
            int(swa_v_head_dim) if swa_v_head_dim is not None else self.v_head_dim
        )
        self.swa_qk_head_dim = (
            int(swa_qk_head_dim)
            if swa_qk_head_dim is not None
            else self.swa_qk_nope_head_dim + self.swa_qk_rope_head_dim
        )

        if mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(1, self.num_hidden_layers) + [
                "sparse"
            ] * max(self.num_hidden_layers - 1, 0)
        else:
            self.mlp_layer_types = list(mlp_layer_types)

        if layer_types is None:
            self.layer_types = _default_layer_types(self.num_hidden_layers)
        else:
            self.layer_types = [_normalize_layer_type(t) for t in layer_types]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(self.layer_types)}) != "
                f"num_hidden_layers ({self.num_hidden_layers})"
            )

        self.index_topk = int(index_topk)
        self.index_head_dim = int(index_head_dim)
        self.index_n_heads = int(index_n_heads)
        self.index_kpool = int(index_kpool)
        self.compressed_indexer_skip_fine_select = bool(
            compressed_indexer_skip_fine_select
        )
        if self.index_kpool < 1:
            raise ValueError(f"index_kpool must be positive, got {self.index_kpool}")
        if self.index_topk % self.index_kpool != 0:
            raise ValueError(
                f"index_topk ({self.index_topk}) must be divisible by "
                f"index_kpool ({self.index_kpool})"
            )

        self.hc_mult = int(hc_mult)
        self.hc_magnitude = float(hc_magnitude)
        self.hc_eps = float(hc_eps)
        self.learnable_sink = bool(learnable_sink)
        self.learnable_sink_init = float(learnable_sink_init)
        self.gated_mla = bool(gated_mla)
        self.attn_post_proj_norm = bool(attn_post_proj_norm)
        self.enable_lm_head_fp32 = bool(enable_lm_head_fp32)
        self.swiglu_limit = float(swiglu_limit)

        self.rope_parameters = _normalize_rope_parameters(
            rope_parameters,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.rope_theta = float(self.rope_parameters[INDEXED_ATTENTION]["rope_theta"])

        for layer_type in set(self.layer_types):
            mrope_section = self.rope_parameters[layer_type]["mrope_section"]
            rope_dim = (
                self.swa_qk_rope_head_dim
                if layer_type == SLIDING_ATTENTION
                else self.qk_rope_head_dim
            )
            if len(mrope_section) != 3 or sum(mrope_section) != rope_dim // 2:
                raise ValueError(
                    f"mrope_section {mrope_section} must have 3 entries summing to "
                    f"{rope_dim // 2} for {layer_type}"
                )

        self.engram_layer_ids = (
            [int(x) for x in engram_layer_ids] if engram_layer_ids is not None else None
        )
        self.engram_max_ngram_size = int(engram_max_ngram_size)
        self.engram_n_head_per_ngram = int(engram_n_head_per_ngram)
        self.engram_n_embed_per_ngram = int(engram_n_embed_per_ngram)
        self.engram_layer_head_vocab_sizes = (
            copy.deepcopy(engram_layer_head_vocab_sizes)
            if engram_layer_head_vocab_sizes is not None
            else None
        )
        self.engram_pad_id = int(engram_pad_id)
        self.engram_fp8_embedding = bool(engram_fp8_embedding)
        self.engram_seed = int(engram_seed)
        self.engram_lookup_table_size = int(engram_lookup_table_size)
        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.scoring_func = str(scoring_func)
        self.topk_method = str(topk_method)

        if self.engram_layer_ids:
            if not (0 <= self.engram_pad_id < self.engram_lookup_table_size):
                raise ValueError(
                    f"engram_pad_id ({self.engram_pad_id}) must lie inside "
                    f"[0, engram_lookup_table_size={self.engram_lookup_table_size})"
                )
            num_orders = self.engram_max_ngram_size - 1
            for layer_id in self.engram_layer_ids:
                sizes = (self.engram_layer_head_vocab_sizes or {}).get(str(layer_id))
                if (
                    not isinstance(sizes, list)
                    or len(sizes) != num_orders
                    or any(
                        not isinstance(order, list)
                        or len(order) != self.engram_n_head_per_ngram
                        for order in sizes
                    )
                ):
                    raise ValueError(
                        f"engram_layer_head_vocab_sizes[{layer_id!r}] must contain "
                        f"{num_orders} lists of {self.engram_n_head_per_ngram} primes"
                    )

        # Published checkpoints store text_config.model_type = "hy_v4_flash";
        # keep the canonical sub-config model_type ("hy_v4_flash_text").
        kwargs.pop("model_type", None)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def paged_cache_layer_types(self) -> list[str]:
        """Cache vocabulary: FULL_ATTENTION for DSA layers, SLIDING_ATTENTION for SWA."""
        return [
            FULL_ATTENTION if t == INDEXED_ATTENTION else SLIDING_ATTENTION
            for t in self.layer_types
        ]

    @property
    def dsa_layer_ids(self) -> tuple[int, ...]:
        return tuple(
            idx
            for idx, layer_type in enumerate(self.layer_types)
            if layer_type == INDEXED_ATTENTION
        )

    @property
    def swa_layer_ids(self) -> tuple[int, ...]:
        return tuple(
            idx
            for idx, layer_type in enumerate(self.layer_types)
            if layer_type == SLIDING_ATTENTION
        )

    def layer_num_attention_heads(self, layer_idx: int) -> int:
        if self.layer_types[layer_idx] == SLIDING_ATTENTION:
            return self.swa_num_attention_heads
        return self.num_attention_heads


class HYV4FlashConfig(PretrainedConfig):
    """Top-level composite configuration for Hy4-flash (vision + text + optional DFly2)."""

    model_type = "hy_v4_flash"
    sub_configs = {
        "vision_config": HYV4FlashVisionConfig,
        "text_config": HYV4FlashTextConfig,
    }

    def __init__(
        self,
        text_config: dict[str, Any] | HYV4FlashTextConfig | None = None,
        vision_config: dict[str, Any] | HYV4FlashVisionConfig | None = None,
        image_token_id: int = 120120,
        video_token_id: int = 120124,
        image_start_token_id: int = 120118,
        image_end_token_id: int = 120119,
        image_newline_token_id: int = 120121,
        tie_word_embeddings: bool = False,
        dfly_num_hidden_layers: int | None = None,
        dfly_block_size: int | None = None,
        dfly_mask_token_id: int | None = None,
        dfly_target_kv_layers: list[int] | None = None,
        dfly_target_index_layer: list[int] | None = None,
        dfly_target_index_topk: int | None = None,
        dfly_mtp_fusion: str | None = None,
        dfly_enable_hidden_correction: bool = False,
        dfly_hidden_correction_intermediate_size: int | None = None,
        dfly_enable_confidence_head: bool = True,
        dfly_attn_causal: bool = True,
        dfly_mlp_type: str = "dense",
        dfly_enable_ihc: bool = False,
        dfly_intermediate_size: int = 9216,
        dfly_context_local_window: int | None = None,
        **kwargs,
    ) -> None:
        if isinstance(vision_config, dict):
            vision_dict = dict(vision_config)
            vision_dict.pop("model_type", None)
            self.vision_config = HYV4FlashVisionConfig(**vision_dict)
        elif vision_config is None:
            self.vision_config = HYV4FlashVisionConfig()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_dict = dict(text_config)
            text_dict.pop("model_type", None)
            text_dict.setdefault("image_token_id", image_token_id)
            text_dict.setdefault("video_token_id", video_token_id)
            self.text_config = HYV4FlashTextConfig(**text_dict)
        elif text_config is None:
            self.text_config = HYV4FlashTextConfig(
                image_token_id=image_token_id,
                video_token_id=video_token_id,
            )
        else:
            self.text_config = text_config

        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.image_start_token_id = int(image_start_token_id)
        self.image_end_token_id = int(image_end_token_id)
        self.image_newline_token_id = int(image_newline_token_id)
        self.text_config.image_token_id = self.image_token_id
        self.text_config.video_token_id = self.video_token_id

        # Embedded DFly2 drafter metadata shipped in published Hy4-flash checkpoints.
        self.dfly_num_hidden_layers = (
            int(dfly_num_hidden_layers) if dfly_num_hidden_layers is not None else None
        )
        self.dfly_block_size = (
            int(dfly_block_size) if dfly_block_size is not None else None
        )
        self.dfly_mask_token_id = (
            int(dfly_mask_token_id) if dfly_mask_token_id is not None else None
        )
        self.dfly_target_kv_layers = (
            [int(x) for x in dfly_target_kv_layers]
            if dfly_target_kv_layers is not None
            else None
        )
        self.dfly_target_index_layer = (
            [int(x) for x in dfly_target_index_layer]
            if dfly_target_index_layer is not None
            else None
        )
        self.dfly_target_index_topk = (
            int(dfly_target_index_topk) if dfly_target_index_topk is not None else None
        )
        self.dfly_mtp_fusion = dfly_mtp_fusion
        self.dfly_enable_hidden_correction = bool(dfly_enable_hidden_correction)
        self.dfly_hidden_correction_intermediate_size = (
            int(dfly_hidden_correction_intermediate_size)
            if dfly_hidden_correction_intermediate_size is not None
            else None
        )
        self.dfly_enable_confidence_head = bool(dfly_enable_confidence_head)
        self.dfly_attn_causal = bool(dfly_attn_causal)
        self.dfly_mlp_type = str(dfly_mlp_type)
        self.dfly_enable_ihc = bool(dfly_enable_ihc)
        self.dfly_intermediate_size = int(dfly_intermediate_size)
        self.dfly_context_local_window = (
            int(dfly_context_local_window)
            if dfly_context_local_window is not None
            else None
        )

        # Forward quantization_config from text_config if top-level omits it.
        if (
            "quantization_config" not in kwargs
            and hasattr(self.text_config, "quantization_config")
            and self.text_config.quantization_config is not None
        ):
            kwargs["quantization_config"] = self.text_config.quantization_config

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class HYV4FlashDFlyConfig(PretrainedConfig):
    """Configuration for the embedded or standalone Hy4-flash DFly2 drafter."""

    model_type = "dfly2"

    def __init__(
        self,
        vocab_size: int = 120832,
        hidden_size: int = 3072,
        intermediate_size: int = 9216,
        num_hidden_layers: int = 5,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        q_lora_rank: int = 1024,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        qk_head_dim: int | None = None,
        rms_norm_eps: float = 1e-5,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        rope_parameters: dict[str, Any] | None = None,
        rope_theta: float = _DEFAULT_INDEXED_ROPE_THETA,
        mask_token_id: int = 120009,
        block_size: int = 6,
        target_kv_layers: list[int] | None = None,
        target_index_layer: list[int] | None = None,
        target_index_topk: int = 2048,
        mtp_fusion: str | None = "slot0",
        enable_hidden_correction: bool = True,
        hidden_correction_intermediate_size: int | None = None,
        enable_confidence_head: bool = True,
        gated_mla: bool = True,
        learnable_sink: bool = True,
        learnable_sink_init: float = 0.0,
        attn_post_proj_norm: bool = True,
        attn_causal: bool = True,
        enable_lm_head_fp32: bool = True,
        **kwargs,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.q_lora_rank = int(q_lora_rank)
        self.kv_lora_rank = int(kv_lora_rank)
        self.qk_nope_head_dim = int(qk_nope_head_dim)
        self.qk_rope_head_dim = int(qk_rope_head_dim)
        self.v_head_dim = int(v_head_dim)
        self.qk_head_dim = (
            int(qk_head_dim)
            if qk_head_dim is not None
            else self.qk_nope_head_dim + self.qk_rope_head_dim
        )
        self.head_dim = self.qk_head_dim
        self.rms_norm_eps = float(rms_norm_eps)
        self.hidden_act = str(hidden_act)
        self.max_position_embeddings = int(max_position_embeddings)

        params = dict(rope_parameters or {})
        params.setdefault("rope_type", "default")
        params.setdefault("rope_theta", float(rope_theta))
        params.setdefault("mrope_section", list(_DEFAULT_MROPE_SECTION))
        params.setdefault("mrope_interleaved", True)
        self.rope_parameters = params
        self.rope_theta = float(params["rope_theta"])

        self.mask_token_id = int(mask_token_id)
        self.block_size = int(block_size)
        self.dfly_block_size = self.block_size
        self.target_kv_layers = (
            [int(x) for x in target_kv_layers]
            if target_kv_layers is not None
            else [25, 29, 33, 37, 41][: self.num_hidden_layers]
        )
        self.target_index_layer = (
            [int(x) for x in target_index_layer]
            if target_index_layer is not None
            else list(self.target_kv_layers)
        )
        self.target_index_topk = int(target_index_topk)
        self.mtp_fusion = mtp_fusion
        self.enable_hidden_correction = bool(enable_hidden_correction)
        self.hidden_correction_intermediate_size = (
            int(hidden_correction_intermediate_size)
            if hidden_correction_intermediate_size is not None
            else self.hidden_size
        )
        self.enable_confidence_head = bool(enable_confidence_head)
        self.gated_mla = bool(gated_mla)
        self.learnable_sink = bool(learnable_sink)
        self.learnable_sink_init = float(learnable_sink_init)
        self.attn_post_proj_norm = bool(attn_post_proj_norm)
        self.attn_causal = bool(attn_causal)
        self.enable_lm_head_fp32 = bool(enable_lm_head_fp32)
        self.sample_from_anchor = False

        kwargs.pop("model_type", None)
        if not kwargs.get("architectures"):
            kwargs["architectures"] = ["HYV4FlashForCausalLMDFly"]
        super().__init__(**kwargs)

    @classmethod
    def from_target(cls, target_config: HYV4FlashConfig) -> HYV4FlashDFlyConfig:
        """Build a DFly2 draft config from an embedded Hy4-flash target checkpoint."""
        text = target_config.text_config
        num_layers = target_config.dfly_num_hidden_layers
        if num_layers is None or num_layers < 1:
            raise ValueError(
                "HYV4FlashConfig does not declare embedded dfly_num_hidden_layers"
            )
        target_kv_layers = target_config.dfly_target_kv_layers
        if target_kv_layers is None or len(target_kv_layers) != num_layers:
            raise ValueError(
                f"dfly_target_kv_layers ({target_kv_layers!r}) must match "
                f"dfly_num_hidden_layers ({num_layers})"
            )
        target_index_layer = target_config.dfly_target_index_layer or list(
            target_kv_layers
        )
        indexed_rope = dict(text.rope_parameters[INDEXED_ATTENTION])
        extra: dict[str, Any] = {}
        quant_cfg = getattr(text, "quantization_config", None) or getattr(
            target_config, "quantization_config", None
        )
        if quant_cfg is not None:
            extra["quantization_config"] = copy.deepcopy(quant_cfg)
        dtype = getattr(text, "dtype", None) or getattr(target_config, "dtype", None)
        if dtype is not None:
            extra["dtype"] = dtype
        return cls(
            vocab_size=text.vocab_size,
            hidden_size=text.hidden_size,
            intermediate_size=target_config.dfly_intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=text.num_attention_heads,
            num_key_value_heads=text.num_attention_heads,
            q_lora_rank=text.q_lora_rank,
            kv_lora_rank=text.kv_lora_rank,
            qk_nope_head_dim=text.qk_nope_head_dim,
            qk_rope_head_dim=text.qk_rope_head_dim,
            v_head_dim=text.v_head_dim,
            qk_head_dim=text.qk_head_dim,
            rms_norm_eps=text.rms_norm_eps,
            hidden_act=text.hidden_act,
            max_position_embeddings=text.max_position_embeddings,
            rope_parameters=indexed_rope,
            rope_theta=float(indexed_rope["rope_theta"]),
            mask_token_id=(
                target_config.dfly_mask_token_id
                if target_config.dfly_mask_token_id is not None
                else 120009
            ),
            block_size=(
                target_config.dfly_block_size
                if target_config.dfly_block_size is not None
                else 6
            ),
            target_kv_layers=list(target_kv_layers),
            target_index_layer=list(target_index_layer),
            target_index_topk=(
                target_config.dfly_target_index_topk
                if target_config.dfly_target_index_topk is not None
                else text.index_topk
            ),
            mtp_fusion=target_config.dfly_mtp_fusion,
            enable_hidden_correction=target_config.dfly_enable_hidden_correction,
            hidden_correction_intermediate_size=(
                target_config.dfly_hidden_correction_intermediate_size
            ),
            enable_confidence_head=target_config.dfly_enable_confidence_head,
            gated_mla=text.gated_mla,
            learnable_sink=text.learnable_sink,
            learnable_sink_init=text.learnable_sink_init,
            attn_post_proj_norm=text.attn_post_proj_norm,
            attn_causal=target_config.dfly_attn_causal,
            enable_lm_head_fp32=text.enable_lm_head_fp32,
            architectures=["HYV4FlashForCausalLMDFly"],
            **extra,
        )


__all__ = [
    "DEEPSEEK_SPARSE_ATTENTION",
    "HYV4FlashConfig",
    "HYV4FlashDFlyConfig",
    "HYV4FlashTextConfig",
    "HYV4FlashVisionConfig",
    "INDEXED_ATTENTION",
]
