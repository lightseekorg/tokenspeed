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

"""Configuration for the Muse Spark language, vision and audio towers."""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"


class MuseSparkConfig(PretrainedConfig):
    """Runtime view of a Muse Spark multimodal checkpoint."""

    model_type = "muse_spark"
    has_no_defaults_at_init = True

    def __init__(
        self,
        text_config: dict,
        vision_config: dict,
        audio_config: dict,
        image_token_id: int,
        video_token_id: int,
        audio_token_id: int,
        out_hidden_size: int,
        projector_hidden_size: int,
        projector_hidden_act: str,
        **kwargs,
    ) -> None:
        text = dict(text_config)
        self.vision_config = dict(vision_config)
        self.audio_config = dict(audio_config)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        self.out_hidden_size = int(out_hidden_size)
        self.projector_hidden_size = int(projector_hidden_size)
        self.projector_hidden_act = projector_hidden_act

        self.vocab_size = int(text["vocab_size"])
        self.hidden_size = int(text["hidden_size"])
        self.head_dim = int(text["head_dim"])
        self.num_attention_heads = int(text["num_attention_heads"])
        self.num_key_value_heads = int(text["num_key_value_heads"])
        self.num_hidden_layers = int(text["num_hidden_layers"])
        self.max_position_embeddings = int(text["max_position_embeddings"])
        self.rms_norm_eps = float(text["rms_norm_eps"])
        self.post_norm_eps = float(text["post_norm_eps"])
        self.final_logit_softcapping = float(text["final_logit_softcapping"])
        self.qk_scale_factor = float(text["qk_scale_factor"])
        self.output_multiplier = float(text["output_multiplier"])
        self.residual_gate_temperature = float(text["residual_gate_temperature"])
        self.moe_hidden_size = int(text["moe_hidden_size"])
        self.moe_intermediate_size = int(text["moe_intermediate_size"])
        self.num_local_experts = int(text["num_local_experts"])
        self.num_experts_per_tok = int(text["num_experts_per_tok"])
        self.num_shared_experts = int(text["num_shared_experts"])
        self.routed_scaling_factor = float(text["routed_scaling_factor"])
        self.layer_types = list(text["layer_types"])
        self.layer_rope_theta = [float(theta) for theta in text["layer_rope_theta"]]
        self.mlp_layer_types = list(text["mlp_layer_types"])
        self.rope_parameters = dict(text["rope_parameters"])
        self.rope_theta = float(self.rope_parameters["rope_theta"])

        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.projector_hidden_act != "gelu":
            raise ValueError("Muse Spark projects vision features through GELU")
        if self.rope_parameters["rope_type"] != "default":
            raise ValueError("Muse Spark requires unscaled decoder RoPE")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must cover every decoder layer")
        if len(self.layer_rope_theta) != self.num_hidden_layers:
            raise ValueError("layer_rope_theta must cover every decoder layer")
        if set(self.layer_types) - {FULL_ATTENTION, SLIDING_ATTENTION}:
            raise ValueError("Muse Spark attends over full or sliding windows")
        for layer_id, layer_type in enumerate(self.layer_types):
            theta = self.layer_rope_theta[layer_id]
            if (theta == 0.0) != (layer_type == FULL_ATTENTION):
                raise ValueError(
                    f"layer {layer_id} is {layer_type} but carries a "
                    f"{theta} RoPE base"
                )
            if theta not in (0.0, self.rope_theta):
                raise ValueError(f"layer {layer_id} disagrees with rope_parameters")
        if set(self.mlp_layer_types) != {"sparse"}:
            raise ValueError("Muse Spark routes experts in every decoder layer")
        if len(self.mlp_layer_types) != self.num_hidden_layers:
            raise ValueError("mlp_layer_types must cover every decoder layer")
        if self.num_shared_experts:
            raise ValueError("Muse Spark supports routed experts only")
        if not 0 < self.num_experts_per_tok <= self.num_local_experts:
            raise ValueError("num_experts_per_tok must be within the expert count")
        if self.routed_scaling_factor <= 0:
            raise ValueError("routed_scaling_factor must be positive")
        if self.hidden_size % self.moe_hidden_size:
            raise ValueError("moe_hidden_size must divide hidden_size")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_key_value_heads must divide num_attention_heads")
        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError("the attention heads must span hidden_size")
        if self.output_multiplier <= 0 or self.qk_scale_factor <= 0:
            raise ValueError("the attention and output multipliers must be positive")

        window = int(text["sliding_window"])
        if window <= 0:
            raise ValueError("sliding_window must be positive")
        self.sliding_window = tuple(
            window if layer_type == SLIDING_ATTENTION else None
            for layer_type in self.layer_types
        )
        full_count = self.layer_types.count(FULL_ATTENTION)
        sliding_count = self.layer_types.count(SLIDING_ATTENTION)
        if full_count and sliding_count:
            subgroup_count = (sliding_count + full_count - 1) // full_count
            sliding_rank = 0
            cache_layer_types = []
            for layer_type in self.layer_types:
                if layer_type == FULL_ATTENTION:
                    cache_layer_types.append(layer_type)
                    continue
                cache_layer_types.append(
                    f"{SLIDING_ATTENTION}_{sliding_rank % subgroup_count}"
                )
                sliding_rank += 1
            self.cache_layer_types = cache_layer_types
        else:
            self.cache_layer_types = list(self.layer_types)

        self.intermediate_size = self.moe_intermediate_size
        self.hidden_act = "silu"
        self.attention_bias = False
        kwargs["architectures"] = ["MuseSparkForConditionalGeneration"]
        kwargs["tie_word_embeddings"] = False
        super().__init__(**kwargs)

    def uses_rope(self, layer_id: int) -> bool:
        """Return whether zero-based decoder layer ``layer_id`` applies interleaved RoPE."""
        return self.layer_rope_theta[layer_id] != 0.0


__all__ = ["FULL_ATTENTION", "SLIDING_ATTENTION", "MuseSparkConfig"]
