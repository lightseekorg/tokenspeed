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

"""
Minimax M3 Model Configuration.
"""

from typing import Any

from tokenspeed.runtime.configs.base_config import BaseConfig, TextConfigBase


class MiniMaxM3VLTextConfig(TextConfigBase):
    model_type = "minimax_m3_vl_text"

    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_gather_output",
        "layers.*.self_attn.k_proj": "colwise_gather_output",
        "layers.*.self_attn.v_proj": "colwise_gather_output",
        "layers.*.self_attn.o_proj": "rowwise_split_input",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

    attribute_map = {
        "num_experts": "num_local_experts",
    }
    default_theta = 5000000.0
    base_config_key = "text_config"

    vocab_size: int = 200064
    hidden_size: int = 6144
    intermediate_size: int = 3072
    num_hidden_layers: int = 60
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 524288
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-06
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 200034
    eos_token_id: int | list[int] | None = 200020
    tie_word_embeddings: bool = False
    attention_dropout: float | int = 0.0
    num_experts_per_tok: int = 4
    num_local_experts: int = 128
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0
    rope_parameters: dict | None = None
    dense_intermediate_size: int = 12288
    shared_intermediate_size: int = 3072
    routed_scaling_factor: float = 2.0
    rotary_dim: int = 64
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    mlp_layer_types: list[str] | None = None
    index_n_heads: int = 4
    index_head_dim: int = 128
    index_block_size: int = 128
    index_topk_blocks: int = 16
    index_local_blocks: int = 1
    layer_types: list[str] | None = None

    def __post_init__(self, **kwargs: Any) -> None:
        sparse_attention_config = kwargs.pop("sparse_attention_config", None)
        moe_layer_freq = kwargs.pop("moe_layer_freq", None)

        super().__post_init__(**kwargs)

        # The checkpoint declares "swigluoai", but the gate is computed inline
        # from swiglu_alpha/limit. The fallback must be a real ACT2FN key.
        self.hidden_act = "silu"

        sparse_cfg = sparse_attention_config or {}
        for flat, legacy in {
            "index_n_heads": "sparse_num_index_heads",
            "index_head_dim": "sparse_index_dim",
            "index_block_size": "sparse_block_size",
            "index_topk_blocks": "sparse_topk_blocks",
            "index_local_blocks": "sparse_local_block",
        }.items():
            if legacy in sparse_cfg:
                setattr(self, flat, sparse_cfg[legacy])

        # `layer_types` is the canonical per-layer attention dispatch: it tells
        # `DynamicCache(config=...)` which layers want the sparse cache and tells
        # `MiniMaxM3VLAttention` which layers build a sparse Lightning Indexer.
        if self.layer_types is None and "sparse_attention_freq" in sparse_cfg:
            self.layer_types = [
                "minimax_m3_sparse" if f else "full_attention"
                for f in sparse_cfg["sparse_attention_freq"]
            ]
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

        # `mlp_layer_types` is the per-layer MLP dispatch read by `MiniMaxM3VLDecoderLayer`:
        if self.mlp_layer_types is None and moe_layer_freq is not None:
            self.mlp_layer_types = ["sparse" if f else "dense" for f in moe_layer_freq]
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["sparse"] * self.num_hidden_layers

        self.validate_layer_type()


class MiniMaxM3VisionConfig(BaseConfig):
    """Configuration for the MiniMax-M3 vision tower.

    Args:
        hidden_size: Vision transformer hidden size.
        intermediate_size: Vision feed-forward intermediate size.
        num_hidden_layers: Number of vision transformer layers.
        num_attention_heads: Number of vision attention heads.
        image_size: Maximum input image size.
        patch_size: Spatial patch size.
        num_channels: Number of image channels.
        temporal_patch_size: Number of frames in each temporal patch.
        spatial_merge_size: Spatial patch-merging factor.
        hidden_act: Vision feed-forward activation.
        layer_norm_eps: Layer normalization epsilon.
        attention_dropout: Vision attention dropout probability.
        rope_parameters: Standard Transformers RoPE parameters.
        initializer_range: Weight initialization standard deviation.
        projection_dim: Legacy checkpoint projection dimension.
        position_embedding_type: Legacy position-embedding selector.
        rope_mode: Legacy rotary position mode.
        rope_theta: Legacy RoPE base, converted into ``rope_parameters``.
        initializer_factor: Legacy initialization multiplier.
        img_token_compression_config: Legacy image-token compression settings.
        vision_segment_max_frames: Maximum frames in one packed vision segment.
        **kwargs: Additional vision checkpoint configuration fields.
    """

    model_type = "minimax_m3"
    base_config_key = "vision_config"

    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    image_size: int = 2016
    patch_size: int = 14
    num_channels: int = 3
    temporal_patch_size: int | None = None
    spatial_merge_size: int | None = None
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    rope_parameters: dict | None = None
    initializer_range: float = 0.02
    projection_dim: int = 6144
    position_embedding_type: str = "rope"
    rope_mode: str = "3d"
    rope_theta: float = 10000.0
    initializer_factor: float = 1.0
    img_token_compression_config: dict | None = None
    vision_segment_max_frames: int | None = 4

    def __post_init__(self, **kwargs) -> None:
        compression_config = dict(self.img_token_compression_config or {})
        if self.temporal_patch_size is None:
            self.temporal_patch_size = int(
                compression_config.get("temporal_patch_size", 2)
            )
        if self.spatial_merge_size is None:
            self.spatial_merge_size = int(
                compression_config.get("spatial_merge_size", 2)
            )
        compression_config.setdefault("image_token_compression_method", "patch_merge")
        compression_config["temporal_patch_size"] = self.temporal_patch_size
        compression_config["spatial_merge_size"] = self.spatial_merge_size

        if self.rope_parameters is None:
            self.rope_parameters = {
                "rope_type": "default",
                "rope_theta": self.rope_theta,
            }
        else:
            self.rope_parameters = dict(self.rope_parameters)
            self.rope_parameters.setdefault("rope_type", "default")
            self.rope_parameters.setdefault("rope_theta", self.rope_theta)

        self.img_token_compression_config = compression_config

        super().__post_init__(**kwargs)


class MiniMaxM3Config(BaseConfig):
    """Combined MiniMax-M3 text and vision configuration.

    Args:
        text_config: Official MiniMax-M3 text config or its dictionary form.
        vision_config: TokenSpeed MiniMax-M3 vision config or its dictionary form.
        image_token_index: Image placeholder token ID.
        video_token_index: Video placeholder token ID.
        image_seq_length: Default number of image tokens.
        process_image_mode: Image preprocessing mode.
        projector_hidden_act: Multimodal projector activation.
        projector_hidden_size: Multimodal projector intermediate size.
        multimodal_projector_bias: Whether projector linear layers use bias.
        vision_feature_layer: Vision layer selected for projection.
        vision_feature_select_strategy: Vision feature selection strategy.
        img_token_compression_config: Outer image-token compression settings.
        image_grid_pinpoints: Dynamic-resolution image grid candidates.
        num_reward_heads: Number of checkpoint reward heads.
        tie_word_embeddings: Whether input and output embeddings are tied.
        **kwargs: Additional outer checkpoint configuration fields.
    """

    model_type = "minimax_m3_vl"
    runtime_attention_arch = "MSA"
    runtime_attention_layer_type = "minimax_m3_sparse"
    sub_configs = {
        "text_config": MiniMaxM3VLTextConfig,
        "vision_config": MiniMaxM3VisionConfig,
    }
    attribute_map = {
        "image_token_id": "image_token_index",
        "video_token_id": "video_token_index",
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: MiniMaxM3VLTextConfig | dict | None = None
    vision_config: MiniMaxM3VisionConfig | dict | None = None
    image_token_index: int = 200025
    video_token_index: int = 200026
    image_seq_length: int = 576
    process_image_mode: str = "dynamic_res"
    projector_hidden_act: str = "gelu"
    projector_hidden_size: int = 6144
    multimodal_projector_bias: bool = True
    vision_feature_layer: int = -1
    vision_feature_select_strategy: str = "full"
    img_token_compression_config: dict | None = None
    image_grid_pinpoints: str | list[tuple[int, int]] | None = None
    num_reward_heads: int = 0
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs) -> None:
        if self.text_config is None:
            self.text_config = MiniMaxM3VLTextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = MiniMaxM3VLTextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = MiniMaxM3VisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = MiniMaxM3VisionConfig(**self.vision_config)

        self.text_config.runtime_attention_layer_type = (
            self.runtime_attention_layer_type
        )
        self.img_token_compression_config = dict(
            self.img_token_compression_config
            or self.vision_config.img_token_compression_config
        )
        self.merged_hidden_size = self.text_config.hidden_size * (
            self.vision_config.spatial_merge_size**2
        )

        if not self.tie_word_embeddings and self.text_config.tie_word_embeddings:
            self.tie_word_embeddings = True

        super().__post_init__(**kwargs)


__all__ = ["MiniMaxM3Config"]
