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

"""Inkling2 checkpoint configuration and separate PE vision/DSpark metadata.

The decoder reuses Inkling's cache geometry, but has a distinct architecture.
Numerical settings come from the checkpoint. MTP metadata does not change the
target's layer count or select the older Inkling NextN execution path.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from transformers.configuration_utils import PretrainedConfig

from tokenspeed.runtime.configs.inkling_config import InklingModelConfig


class Inkling2ModelConfig(InklingModelConfig):
    """Decoder settings, with explicit checkpoint numerical parameters.

    The keyword arguments contain the decoder geometry and routing settings.
    Missing checkpoint fields fail instead of selecting reference defaults.
    """

    model_type = "inkling2_model"
    has_no_defaults_at_init = True

    def __init__(
        self,
        *,
        model_max_length: int,
        use_value_norm_no_gamma: bool,
        qk_l2_clip_threshold: float,
        rel_l2_clip_threshold: float,
        n_layers_per_attn_res_block: int,
        attn_res_normalize_values: bool,
        mlp_gate_softcap: float,
        mlp_up_softcap: float,
        **kwargs: Any,
    ) -> None:
        required = {
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "d_rel",
            "rel_extent",
            "local_layer_ids",
            "sliding_window_size",
            "swa_num_attention_heads",
            "swa_num_key_value_heads",
            "swa_head_dim",
            "rms_norm_eps",
            "q_bias",
            "o_bias",
            "sconv_kernel_size",
            "n_routed_experts",
            "n_shared_experts",
            "num_experts_per_tok",
            "route_scale",
            "use_gate_bias",
            "log_scaling_n_floor",
            "log_scaling_alpha",
            "unpadded_vocab_size",
            "final_logit_softcapping",
        }
        missing = required - kwargs.keys()
        if missing:
            raise ValueError(f"Inkling2 text_config is missing {sorted(missing)}")
        if (
            "max_position_embeddings" in kwargs
            and kwargs["max_position_embeddings"] != model_max_length
        ):
            raise ValueError("max_position_embeddings must match model_max_length")
        for value_dim, key_dim in (
            ("v_head_dim", "head_dim"),
            ("swa_v_head_dim", "swa_head_dim"),
        ):
            if value_dim in kwargs and kwargs[value_dim] != kwargs[key_dim]:
                raise ValueError(f"{value_dim} must match {key_dim}")
        fixed = {
            "use_embed_norm": True,
            "use_sconv": True,
            "dense_mlp_idx": 0,
            "use_global_scale": False,
            "norm_after_topk": True,
            "gate_activation": "sigmoid",
            "shared_expert_sink": True,
            "inference_moe_w13_interleaved": True,
            "num_nextn_predict_layers": 0,
            "tie_word_embeddings": False,
            "hidden_act": "silu",
            "pre_norm_pre_affine": False,
            "final_norm_pre_affine": False,
        }
        for name, value in fixed.items():
            if name in kwargs and kwargs[name] != value:
                raise ValueError(f"Inkling2 requires {name}={value!r}")
            kwargs[name] = value
        super().__init__(model_max_length=model_max_length, **kwargs)
        self.max_position_embeddings = model_max_length
        self.use_value_norm_no_gamma = use_value_norm_no_gamma
        self.qk_l2_clip_threshold = qk_l2_clip_threshold
        self.rel_l2_clip_threshold = rel_l2_clip_threshold
        self.n_layers_per_attn_res_block = n_layers_per_attn_res_block
        self.attn_res_normalize_values = attn_res_normalize_values
        self.mlp_gate_softcap = mlp_gate_softcap
        self.mlp_up_softcap = mlp_up_softcap
        if not 0 < n_layers_per_attn_res_block <= self.num_hidden_layers:
            raise ValueError(
                "Attention residual block size must be within the layer count"
            )
        if not 0 < self.num_experts_per_tok <= self.n_routed_experts:
            raise ValueError(
                "num_experts_per_tok must be within the routed expert count"
            )
        if self.n_shared_experts < 0:
            raise ValueError("n_shared_experts must be nonnegative")
        if (
            self.num_key_value_heads <= 0
            or self.num_attention_heads % self.num_key_value_heads
        ):
            raise ValueError("Query heads must be divisible by positive KV heads")
        if self.ckpt_num_key_value_heads != self.swa_num_key_value_heads:
            raise ValueError(
                "Inkling2 requires matching full and sliding KV-head counts"
            )
        for name, value in (
            ("qk_l2_clip_threshold", qk_l2_clip_threshold),
            ("rel_l2_clip_threshold", rel_l2_clip_threshold),
            ("route_scale", self.route_scale),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("mlp_gate_softcap", mlp_gate_softcap),
            ("mlp_up_softcap", mlp_up_softcap),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


class Inkling2VisionConfig(PretrainedConfig):
    """PE vision geometry. This is not the older Inkling HMLP encoder."""

    model_type = "inkling2_vision_model"
    has_no_defaults_at_init = True

    def __init__(
        self,
        *,
        temporal_patch_size: int,
        native_patch_size: int,
        adapter_merge_size: int,
        projection_hidden_dim: int,
        projector_bias: bool,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        mlp_ratio: float,
        posemb_grid_size: int,
        layer_norm_eps: float,
        **kwargs: Any,
    ) -> None:
        self.temporal_patch_size = temporal_patch_size
        self.native_patch_size = native_patch_size
        self.adapter_merge_size = adapter_merge_size
        self.projection_hidden_dim = projection_hidden_dim
        self.projector_bias = projector_bias
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.posemb_grid_size = posemb_grid_size
        self.layer_norm_eps = layer_norm_eps
        super().__init__(**kwargs)


class Inkling2MMConfig(PretrainedConfig):
    """Container for independent decoder, PE vision, and DSpark settings.

    ``text_config`` and ``vision_config`` accept configuration objects or
    checkpoint dictionaries. ``mtp_config=None`` declares no draft model.
    """

    model_type = "inkling2"
    has_no_defaults_at_init = True
    sub_configs = {
        "text_config": Inkling2ModelConfig,
        "vision_config": Inkling2VisionConfig,
    }

    def __init__(
        self,
        *,
        text_config: dict[str, Any] | Inkling2ModelConfig,
        vision_config: dict[str, Any] | Inkling2VisionConfig | None,
        mtp_config: dict[str, Any] | None,
        eos_token_id: int | list[int],
        **kwargs: Any,
    ) -> None:
        self.text_config = (
            text_config
            if isinstance(text_config, Inkling2ModelConfig)
            else Inkling2ModelConfig(**copy.deepcopy(text_config))
        )
        self.vision_config = (
            vision_config
            if isinstance(vision_config, Inkling2VisionConfig) or vision_config is None
            else Inkling2VisionConfig(**copy.deepcopy(vision_config))
        )
        self.mtp_config = copy.deepcopy(mtp_config)
        self.text_config.eos_token_id = eos_token_id
        if kwargs.pop("tie_word_embeddings", False):
            raise ValueError("Inkling2 requires tie_word_embeddings=False")
        super().__init__(tie_word_embeddings=False, eos_token_id=eos_token_id, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def num_hidden_layers(self) -> int:
        return self.text_config.num_hidden_layers

    @property
    def cache_layer_types(self) -> list[str]:
        return self.text_config.cache_layer_types

    @property
    def sliding_window(self) -> int:
        return self.text_config.sliding_window

    @property
    def image_placeholder_token_id(self) -> int | None:
        """The native image payload token, not the transport's soft anchor."""
        return 200005 if self.vision_config is not None else None


class Inkling2MTPConfig(PretrainedConfig):
    """Independent RoPE/GQA DSpark draft configuration, not Inkling NextN."""

    model_type = "inkling2_mtp"
    has_no_defaults_at_init = True

    def __init__(
        self,
        *,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        max_position_embeddings: int,
        q_bias: bool,
        o_bias: bool,
        rms_norm_eps: float,
        rope_theta: float,
        local_layer_ids: list[int],
        sliding_window_size: int,
        block_size: int,
        markov_rank: int,
        markov_head_type: str,
        target_layer_ids: list[int],
        **kwargs: Any,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.q_bias = q_bias
        self.o_bias = o_bias
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.local_layer_ids = list(local_layer_ids)
        self.sliding_window_size = sliding_window_size
        self.block_size = block_size
        self.markov_rank = markov_rank
        self.markov_head_type = markov_head_type
        self.target_layer_ids = list(target_layer_ids)
        if (
            num_hidden_layers <= 0
            or len(set(local_layer_ids)) != len(local_layer_ids)
            or any(i < 0 or i >= num_hidden_layers for i in local_layer_ids)
        ):
            raise ValueError(
                "MTP local_layer_ids must contain unique valid layer indices"
            )
        if markov_head_type != "rnn_hidden":
            raise ValueError("Inkling2 MTP requires markov_head_type='rnn_hidden'")
        if markov_rank <= 0 or block_size < 2:
            raise ValueError("Inkling2 MTP needs positive rank and block_size >= 2")
        if num_key_value_heads <= 0 or num_attention_heads % num_key_value_heads:
            raise ValueError("MTP query heads must be divisible by positive KV heads")
        if "dtype" not in kwargs and "torch_dtype" not in kwargs:
            raise ValueError("Inkling2 MTP requires explicit dtype or torch_dtype")
        if any(
            str(kwargs[key]) not in ("bfloat16", "torch.bfloat16")
            for key in ("dtype", "torch_dtype")
            if key in kwargs
        ):
            raise ValueError("Inkling2 MTP requires bfloat16 precision")
        if kwargs.pop("quantization_config", None) not in (None, {}):
            raise ValueError("Inkling2 MTP requires unquantized BF16 weights")
        # An explicit empty config skips the target manifest in a shared checkpoint.
        super().__init__(quantization_config={}, **kwargs)

    @classmethod
    def from_target(cls, config: Inkling2MMConfig) -> Inkling2MTPConfig:
        """Build draft metadata from a target without changing target settings.

        Args:
            config: Target container with its checkpoint ``mtp_config``.

        Returns:
            A distinct DSpark draft configuration with target feature taps.
        """
        if config.mtp_config is None:
            raise ValueError("Inkling2 speculative decoding requires mtp_config")
        text = config.text_config
        draft = cls(**copy.deepcopy(config.mtp_config))
        if draft.hidden_size != text.hidden_size:
            raise ValueError("Inkling2 MTP and target hidden sizes must match")
        features = draft.target_layer_ids
        if (
            len(features) < 2
            or features != sorted(set(features))
            or (features[-1] != text.num_hidden_layers)
            or any(i < 0 or i >= text.num_hidden_layers for i in features[:-1])
        ):
            raise ValueError(
                "MTP target_layer_ids must be sorted unique target indices "
                "followed by the final-norm sentinel num_hidden_layers"
            )
        values = draft.to_dict()
        values.update(
            architectures=["Inkling2ForConditionalGenerationMTP"],
            vocab_size=text.vocab_size,
            unpadded_vocab_size=text.unpadded_vocab_size,
            padded_vocab_size=text.padded_vocab_size,
            target_num_hidden_layers=text.num_hidden_layers,
            final_logit_softcapping=text.final_logit_softcapping,
            hidden_act="silu",
            tie_word_embeddings=False,
            is_causal=False,
            sample_from_anchor=False,
            is_hybrid_swa=bool(draft.local_layer_ids),
            hybrid_layer_pattern=[
                int(i in draft.local_layer_ids) for i in range(draft.num_hidden_layers)
            ],
            layer_types=[
                "sliding_attention" if i in draft.local_layer_ids else "full_attention"
                for i in range(draft.num_hidden_layers)
            ],
            sliding_window=draft.sliding_window_size,
            dspark_block_size=draft.block_size - 1,
            dspark_target_layer_ids=list(features),
            mask_token_id=0,
            dflash_config=dict(
                num_target_layers=text.num_hidden_layers,
                target_layer_ids=list(features[:-1]),
            ),
        )
        return cls(**values)
