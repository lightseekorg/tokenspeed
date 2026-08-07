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

"""Qwen3-ASR nested configuration definitions.

Transformers 5.12 contains the shared Qwen3-Omni audio architecture but does
not yet register ``model_type: qwen3_asr``.  These small wrappers preserve the
official checkpoint's ``thinker_config`` shape while reusing TokenSpeed's
Qwen3 text configuration.
"""

from __future__ import annotations

from typing import Any

from tokenspeed.runtime.configs.base_config import BaseConfig
from tokenspeed.runtime.configs.qwen3_config import Qwen3Config


class Qwen3ASRAudioEncoderConfig(BaseConfig):
    """Qwen3 audio tower configuration shipped by Qwen3-ASR."""

    model_type = "qwen3_asr_audio_encoder"
    base_config_key = "audio_config"
    attribute_map = {
        "num_hidden_layers": "encoder_layers",
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "intermediate_size": "encoder_ffn_dim",
    }

    num_mel_bins: int = 128
    encoder_layers: int = 24
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    d_model: int = 1024
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_function: str = "gelu"
    activation_dropout: float = 0.0
    scale_embedding: bool = False
    initializer_range: float = 0.02
    max_source_positions: int = 1500
    n_window: int = 50
    n_window_infer: int = 800
    conv_chunksize: int = 500
    downsample_hidden_size: int = 480
    output_dim: int = 2048

    def __post_init__(self, **kwargs: Any) -> None:
        super().__post_init__(**kwargs)


class Qwen3ASRThinkerConfig(BaseConfig):
    """Audio/text thinker configuration nested below the outer ASR config."""

    model_type = "qwen3_asr_thinker"
    base_config_key = "thinker_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {
        "audio_config": Qwen3ASRAudioEncoderConfig,
        "text_config": Qwen3Config,
    }

    audio_config: dict[str, Any] | Qwen3ASRAudioEncoderConfig | None = None
    text_config: dict[str, Any] | Qwen3Config | None = None
    audio_start_token_id: int = 151669
    audio_end_token_id: int = 151670
    audio_token_id: int = 151676
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs: Any) -> None:
        self.audio_config = self._ensure_audio_config(self.audio_config)
        self.text_config = self._ensure_text_config(self.text_config)
        super().__post_init__(**kwargs)

    @staticmethod
    def _ensure_audio_config(
        config: dict[str, Any] | Qwen3ASRAudioEncoderConfig | None,
    ) -> Qwen3ASRAudioEncoderConfig:
        if isinstance(config, Qwen3ASRAudioEncoderConfig):
            return config
        return Qwen3ASRAudioEncoderConfig(**(config or {}))

    @staticmethod
    def _ensure_text_config(
        config: dict[str, Any] | Qwen3Config | None,
    ) -> Qwen3Config:
        if isinstance(config, Qwen3Config):
            return config
        return Qwen3Config(**(config or {}))

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "audio_config" and isinstance(value, dict):
            value = self._ensure_audio_config(value)
        elif name == "text_config" and isinstance(value, dict):
            value = self._ensure_text_config(value)
        super().__setattr__(name, value)

    def get_text_config(self, *args: Any, **kwargs: Any) -> Qwen3Config:
        return self.text_config


class Qwen3ASRConfig(BaseConfig):
    """Outer config matching ``Qwen/Qwen3-ASR-*`` config.json files."""

    model_type = "qwen3_asr"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"thinker_config": Qwen3ASRThinkerConfig}

    thinker_config: dict[str, Any] | Qwen3ASRThinkerConfig | None = None
    support_languages: list[str] | None = None

    def __post_init__(self, **kwargs: Any) -> None:
        self.thinker_config = self._ensure_thinker_config(self.thinker_config)
        self.support_languages = list(self.support_languages or [])
        super().__post_init__(**kwargs)

    @staticmethod
    def _ensure_thinker_config(
        config: dict[str, Any] | Qwen3ASRThinkerConfig | None,
    ) -> Qwen3ASRThinkerConfig:
        if isinstance(config, Qwen3ASRThinkerConfig):
            return config
        return Qwen3ASRThinkerConfig(**(config or {}))

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "thinker_config" and isinstance(value, dict):
            value = self._ensure_thinker_config(value)
        super().__setattr__(name, value)

    def get_text_config(self, *args: Any, **kwargs: Any) -> Qwen3Config:
        return self.thinker_config.text_config

    @property
    def audio_config(self) -> Qwen3ASRAudioEncoderConfig:
        return self.thinker_config.audio_config

    @property
    def audio_start_token_id(self) -> int:
        return self.thinker_config.audio_start_token_id

    @property
    def audio_end_token_id(self) -> int:
        return self.thinker_config.audio_end_token_id

    @property
    def audio_token_id(self) -> int:
        return self.thinker_config.audio_token_id


__all__ = [
    "Qwen3ASRAudioEncoderConfig",
    "Qwen3ASRConfig",
    "Qwen3ASRThinkerConfig",
]
