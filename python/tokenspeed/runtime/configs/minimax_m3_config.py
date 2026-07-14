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

"""MiniMax-M3 multimodal and text configuration definitions."""

from transformers.configuration_utils import PretrainedConfig


class MiniMaxM3TextConfig(PretrainedConfig):
    """Configuration for the MiniMax-M3 language model."""

    model_type = "minimax_m3_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 200064,
        hidden_size: int = 6144,
        intermediate_size: int = 3072,
        dense_intermediate_size: int = 12288,
        shared_intermediate_size: int = 3072,
        num_hidden_layers: int = 60,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        max_position_embeddings: int = 1048576,
        rms_norm_eps: float = 1e-6,
        use_gemma_norm: bool = True,
        attention_output_gate: bool = False,
        rope_theta: float = 5_000_000,
        rotary_dim: int = 64,
        partial_rotary_factor: float = 0.5,
        hidden_act: str = "swigluoai",
        use_qk_norm: bool = True,
        qk_norm_type: str = "per_head",
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        num_local_experts: int = 128,
        num_experts_per_tok: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "sigmoid",
        use_routing_bias: bool = True,
        routed_scaling_factor: float = 2.0,
        moe_layer_freq: list[int] | None = None,
        swiglu_alpha: float = 1.702,
        swiglu_limit: float = 7.0,
        sparse_attention_config: dict | None = None,
        num_mtp_modules: int = 1,
        **kwargs,
    ) -> None:
        if moe_layer_freq is None:
            moe_layer_freq = [0, 0, 0] + [1] * (num_hidden_layers - 3)
        if len(moe_layer_freq) != num_hidden_layers:
            raise ValueError(
                "moe_layer_freq must have one entry per decoder layer: "
                f"got {len(moe_layer_freq)} entries for {num_hidden_layers} layers."
            )

        if sparse_attention_config is None:
            sparse_attention_config = {
                "use_sparse_attention": True,
                "sparse_index_dim": 128,
                "sparse_num_index_heads": 4,
                "sparse_topk_blocks": 16,
                "sparse_block_size": 128,
                "sparse_disable_index_value": list(moe_layer_freq),
                "sparse_score_type": "max",
                "sparse_init_block": 0,
                "sparse_local_block": 1,
                "sparse_attention_freq": list(moe_layer_freq),
            }

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dense_intermediate_size = dense_intermediate_size
        self.shared_intermediate_size = shared_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_gemma_norm = use_gemma_norm
        self.attention_output_gate = attention_output_gate
        self.rope_theta = rope_theta
        self.rotary_dim = rotary_dim
        self.partial_rotary_factor = partial_rotary_factor
        self.hidden_act = hidden_act
        self.use_qk_norm = use_qk_norm
        self.qk_norm_type = qk_norm_type
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.use_routing_bias = use_routing_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_layer_freq = moe_layer_freq
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        self.sparse_attention_config = sparse_attention_config
        self.num_mtp_modules = num_mtp_modules

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class MiniMaxM3VisionConfig(PretrainedConfig):
    """Configuration container for the MiniMax-M3 vision tower."""

    model_type = "clip_vision_model"
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
        projection_dim: int = 6144,
        position_embedding_type: str = "rope",
        rope_mode: str = "3d",
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        hidden_act: str = "gelu",
        initializer_factor: float = 1.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        img_token_compression_config: dict | None = None,
        vision_segment_max_frames: int = 4,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.projection_dim = projection_dim
        self.position_embedding_type = position_embedding_type
        self.rope_mode = rope_mode
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.initializer_factor = initializer_factor
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.img_token_compression_config = img_token_compression_config or {
            "image_token_compression_method": "patch_merge",
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        }
        self.vision_segment_max_frames = vision_segment_max_frames
        super().__init__(**kwargs)


class MiniMaxM3VLConfig(PretrainedConfig):
    """Configuration for MiniMax-M3's vision-language wrapper."""

    model_type = "minimax_m3_vl"
    sub_configs = {
        "vision_config": MiniMaxM3VisionConfig,
        "text_config": MiniMaxM3TextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: MiniMaxM3TextConfig | dict | None = None,
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
        image_grid_pinpoints: str | None = None,
        num_reward_heads: int = 0,
        **kwargs,
    ) -> None:
        self.text_config = self._make_text_config(text_config)
        self.vision_config = self._make_vision_config(vision_config)
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.image_seq_length = image_seq_length
        self.process_image_mode = process_image_mode
        self.projector_hidden_act = projector_hidden_act
        self.projector_hidden_size = projector_hidden_size
        self.multimodal_projector_bias = multimodal_projector_bias
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.img_token_compression_config = img_token_compression_config or {}
        self.image_grid_pinpoints = image_grid_pinpoints
        self.num_reward_heads = num_reward_heads
        super().__init__(**kwargs)

    @staticmethod
    def _make_text_config(
        config: MiniMaxM3TextConfig | dict | None,
    ) -> MiniMaxM3TextConfig:
        if isinstance(config, MiniMaxM3TextConfig):
            return config
        return MiniMaxM3TextConfig(**(config or {}))

    @staticmethod
    def _make_vision_config(
        config: MiniMaxM3VisionConfig | dict | None,
    ) -> MiniMaxM3VisionConfig:
        if isinstance(config, MiniMaxM3VisionConfig):
            return config
        return MiniMaxM3VisionConfig(**(config or {}))

    def __setattr__(self, name, value):
        if name == "text_config" and isinstance(value, dict):
            value = self._make_text_config(value)
        elif name == "vision_config" and isinstance(value, dict):
            value = self._make_vision_config(value)
        super().__setattr__(name, value)


__all__ = ["MiniMaxM3TextConfig", "MiniMaxM3VisionConfig", "MiniMaxM3VLConfig"]
