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

from transformers.configuration_utils import PretrainedConfig
from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
    MiniMaxM3VLTextConfig,
)


class MiniMaxM3VisionConfig(PretrainedConfig):
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

    def __init__(
        self,
        hidden_size: int = 1280,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 16,
        image_size: int = 2016,
        patch_size: int = 14,
        num_channels: int = 3,
        temporal_patch_size: int | None = None,
        spatial_merge_size: int | None = None,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        rope_parameters: dict | None = None,
        initializer_range: float = 0.02,
        projection_dim: int = 6144,
        position_embedding_type: str = "rope",
        rope_mode: str = "3d",
        rope_theta: float = 10000.0,
        initializer_factor: float = 1.0,
        img_token_compression_config: dict | None = None,
        vision_segment_max_frames: int | None = 4,
        **kwargs,
    ) -> None:
        compression_config = dict(img_token_compression_config or {})
        if temporal_patch_size is None:
            temporal_patch_size = int(compression_config.get("temporal_patch_size", 2))
        if spatial_merge_size is None:
            spatial_merge_size = int(compression_config.get("spatial_merge_size", 2))
        compression_config.setdefault("image_token_compression_method", "patch_merge")
        compression_config["temporal_patch_size"] = temporal_patch_size
        compression_config["spatial_merge_size"] = spatial_merge_size

        if rope_parameters is None:
            rope_parameters = {
                "rope_type": "default",
                "rope_theta": rope_theta,
            }
        else:
            rope_parameters = dict(rope_parameters)
            rope_parameters.setdefault("rope_type", "default")
            rope_parameters.setdefault("rope_theta", rope_theta)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters
        self.rope_theta = rope_theta
        self.initializer_range = initializer_range
        self.projection_dim = projection_dim
        self.position_embedding_type = position_embedding_type
        self.rope_mode = rope_mode
        self.initializer_factor = initializer_factor
        self.img_token_compression_config = compression_config
        self.vision_segment_max_frames = vision_segment_max_frames
        super().__init__(**kwargs)


class MiniMaxM3Config(PretrainedConfig):
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

    def __init__(
        self,
        text_config: MiniMaxM3VLTextConfig | dict | None = None,
        vision_config: MiniMaxM3VisionConfig | dict | None = None,
        image_token_index: int = 200025,
        video_token_index: int = 200026,
        image_seq_length: int = 576,
        process_image_mode: str = "dynamic_res",
        projector_hidden_act: str = "gelu",
        projector_hidden_size: int = 6144,
        multimodal_projector_bias: bool = True,
        vision_feature_layer: int = -1,
        vision_feature_select_strategy: str = "full",
        img_token_compression_config: dict | None = None,
        image_grid_pinpoints: str | list[tuple[int, int]] | None = None,
        num_reward_heads: int = 0,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        if text_config is None:
            text_config = MiniMaxM3VLTextConfig()
        elif isinstance(text_config, dict):
            text_values = dict(text_config)
            text_values.pop("model_type", None)
            text_config = MiniMaxM3VLTextConfig(**text_values)

        if vision_config is None:
            vision_config = MiniMaxM3VisionConfig()
        elif isinstance(vision_config, dict):
            vision_values = dict(vision_config)
            vision_values.pop("model_type", None)
            vision_config = MiniMaxM3VisionConfig(**vision_values)

        self.text_config = text_config
        self.text_config.runtime_attention_layer_type = (
            self.runtime_attention_layer_type
        )
        self.vision_config = vision_config
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.image_seq_length = image_seq_length
        self.process_image_mode = process_image_mode
        self.projector_hidden_act = projector_hidden_act
        self.projector_hidden_size = projector_hidden_size
        self.multimodal_projector_bias = multimodal_projector_bias
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.img_token_compression_config = dict(
            img_token_compression_config or vision_config.img_token_compression_config
        )
        self.image_grid_pinpoints = image_grid_pinpoints
        self.num_reward_heads = num_reward_heads
        self.merged_hidden_size = text_config.hidden_size * (
            vision_config.spatial_merge_size**2
        )

        if not tie_word_embeddings and text_config.tie_word_embeddings:
            tie_word_embeddings = True
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["MiniMaxM3Config"]
