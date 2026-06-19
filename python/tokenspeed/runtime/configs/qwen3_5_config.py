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

"""Qwen 3.5 configuration wrappers."""

from transformers import PretrainedConfig

from tokenspeed.runtime.configs.qwen3_5_text_base_config import Qwen3_5BaseTextConfig
from tokenspeed.runtime.configs.qwen3_vision_config import Qwen3VLVisionConfig

_MROPE_EXTENSION_KEYS = frozenset({"mrope_section", "mrope_interleaved"})


def _to_transformers_rope_parameters(rope_config):
    if not isinstance(rope_config, dict):
        return rope_config
    return {
        key: value
        for key, value in rope_config.items()
        if key not in _MROPE_EXTENSION_KEYS
    }


class Qwen3_5VisionConfig(Qwen3VLVisionConfig):
    model_type = "qwen3_5"
    base_config_key = "vision_config"


class Qwen3_5TextConfig(Qwen3_5BaseTextConfig):
    model_type = "qwen3_5_text"
    base_config_key = "text_config"

    def __init__(
        self,
        **kwargs,
    ):
        # HF Qwen3.5 checkpoints may provide RoPE settings under rope_parameters.
        # Normalize it before parent init so downstream code sees the expected values.
        full_rope_parameters = kwargs.get("rope_parameters")
        if full_rope_parameters is not None:
            kwargs["rope_parameters"] = _to_transformers_rope_parameters(
                full_rope_parameters
            )

        super().__init__(**kwargs)
        self._tokenspeed_rope_parameters = full_rope_parameters or self.rope_parameters


class Qwen3_5Config(PretrainedConfig):
    model_type = "qwen3_5"
    sub_configs = {
        "vision_config": Qwen3_5VisionConfig,
        "text_config": Qwen3_5TextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vision_config = self._ensure_vision_config(vision_config)
        self.text_config = self._ensure_text_config(text_config)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)

    def _ensure_text_config(self, text_config):
        """Convert text_config to the proper config class if it's a dict."""
        text_cls = self.sub_configs["text_config"]
        if isinstance(text_config, dict):
            return text_cls(**text_config)
        if isinstance(text_config, Qwen3_5Config):
            nested_text_config = text_config.__dict__.get("text_config")
            if nested_text_config is not None and nested_text_config is not text_config:
                return self._ensure_text_config(nested_text_config)
        if text_config is None:
            return text_cls()
        return text_config

    def _ensure_vision_config(self, vision_config):
        """Convert vision_config to the proper config class if it's a dict."""
        vision_cls = self.sub_configs["vision_config"]
        if isinstance(vision_config, dict):
            return vision_cls(**vision_config)
        if vision_config is None:
            return vision_cls()
        return vision_config

    def __setattr__(self, name, value):
        # from_pretrained re-assigns text_config as a raw dict after __init__;
        # intercept and convert it back to the proper config class.
        if name == "text_config" and isinstance(value, dict):
            value = self._ensure_text_config(value)
        elif name == "vision_config" and isinstance(value, dict):
            value = self._ensure_vision_config(value)
        super().__setattr__(name, value)

    def __getattr__(self, name):
        """Forward attribute access to text_config for inference-only usage."""
        if name.startswith("_") or name in {"text_config", "vision_config"}:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        text_config = self.__dict__.get("text_config")
        if isinstance(text_config, dict):
            text_config = self._ensure_text_config(text_config)
            self.__dict__["text_config"] = text_config
        if text_config is None or text_config is self:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        try:
            return getattr(text_config, name)
        except AttributeError as exc:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'"
            ) from exc


class Qwen3_5MoeVisionConfig(Qwen3_5VisionConfig):
    model_type = "qwen3_5_moe"


class Qwen3_5MoeTextConfig(Qwen3_5TextConfig):
    model_type = "qwen3_5_moe_text"

    def __init__(self, **kwargs):
        # Explicit __init__ prevents transformers from auto-generating one
        # that skips Qwen3_5TextConfig.__init__ (rope_parameters normalization).
        super().__init__(**kwargs)


class Qwen3_5MoeConfig(Qwen3_5Config):
    model_type = "qwen3_5_moe"
    sub_configs = {
        "vision_config": Qwen3_5MoeVisionConfig,
        "text_config": Qwen3_5MoeTextConfig,
    }

    def __init__(self, **kwargs):
        # Explicit __init__ prevents transformers from auto-generating one
        # that skips Qwen3_5Config.__init__ (text/vision config setup).
        super().__init__(**kwargs)
