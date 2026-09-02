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

"""Qwen4-Exp text and multimodal configuration definitions."""

from __future__ import annotations

from tokenspeed.runtime.configs.qwen3_5_config import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION


class Qwen4ExpVisionConfig(Qwen3_5VisionConfig):
    """Vision-tower configuration embedded by a Qwen4-Exp checkpoint."""

    model_type = "qwen4_exp"
    base_config_key = "vision_config"


class Qwen4ExpTextConfig(Qwen3_5TextConfig):
    """Text configuration for the Qwen4-Exp hybrid decoder.

    Qwen4-Exp extends the Qwen3.5 GDN/full-attention layout with a widened
    hyper-connection residual stream and optional PLE/QSA components.
    """

    model_type = "qwen4_exp_text"
    base_config_key = "text_config"

    def __init__(
        self,
        hc_count: int = 4,
        hc_lowrank: int = 320,
        ple_layer_ids: list[int] | None = None,
        ple_embed_dim: int | None = None,
        ple_conv_kernel_size: int = 4,
        ple_embed_dtype: str | None = None,
        ple_offload_embedding: bool = True,
        ngram_size: int = 3,
        heads_per_ngram: int = 8,
        ngram_vocab_size_base: int = 20_000_000,
        make_ngram_vocab_size_divisible_by: int = 128,
        layer_types: list[str] | None = None,
        rope_parameters: dict | None = None,
        num_experts: int | None = None,
        **kwargs,
    ) -> None:
        if hc_count <= 1:
            raise ValueError(f"Qwen4-Exp requires hc_count > 1, got {hc_count}.")
        if rope_parameters is not None:
            kwargs.setdefault("rope_parameters", rope_parameters)
        super().__init__(
            layer_types=layer_types,
            num_experts=num_experts,
            **kwargs,
        )
        self.hc_count = int(hc_count)
        self.hc_lowrank = int(hc_lowrank)
        self.ple_layer_ids = list(ple_layer_ids or [])
        self.ple_embed_dim = int(ple_embed_dim or self.hidden_size)
        self.ple_conv_kernel_size = int(ple_conv_kernel_size)
        # Storage dtype for the PLE n-gram embedding table. None keeps the
        # model dtype; "float8_e4m3fn" stores the table in FP8 with online
        # per-row quantization at load time (halves the table's memory).
        self.ple_embed_dtype = ple_embed_dtype
        # Keep the n-gram table in page-locked host memory and let the gather
        # kernel read it over PCIe/C2C. The table scales with
        # ngram_vocab_size_base and does not fit in device memory at production
        # sizes; host residency trades interconnect bandwidth for capacity.
        self.ple_offload_embedding = bool(ple_offload_embedding)
        self.ngram_size = int(ngram_size)
        self.heads_per_ngram = int(heads_per_ngram)
        self.ngram_vocab_size_base = int(ngram_vocab_size_base)
        self.make_ngram_vocab_size_divisible_by = int(
            make_ngram_vocab_size_divisible_by
        )
        self._qwen4_exp_layer_types = (
            list(layer_types) if layer_types is not None else None
        )

    @property
    def layers_block_type(self) -> list[str]:
        if self._qwen4_exp_layer_types is None:
            return super().layers_block_type
        return [
            "attention" if layer_type == FULL_ATTENTION else layer_type
            for layer_type in self._qwen4_exp_layer_types
        ]

    @property
    def layer_types(self) -> list[str]:
        if self._qwen4_exp_layer_types is None:
            return super().layer_types
        return [
            FULL_ATTENTION if layer_type == "attention" else layer_type
            for layer_type in self._qwen4_exp_layer_types
        ]

    @layer_types.setter
    def layer_types(self, value: list[str] | None) -> None:
        # Qwen3_5BaseTextConfig assigns this name during parent construction;
        # retain the serialized list without shadowing the normalized property.
        self._qwen4_exp_layer_types = list(value) if value is not None else None

    @property
    def short_conv_layer_ids(self) -> list[int]:
        return sorted({int(layer_id) - 1 for layer_id in self.ple_layer_ids})

    @property
    def short_conv_state_shape(self) -> tuple[int, int] | None:
        if not self.short_conv_layer_ids:
            return None
        state_len = (self.ple_conv_kernel_size - 1) * self.ngram_size
        return self.hidden_size * self.hc_count, state_len

    @property
    def ngram_context_len(self) -> int:
        return max(self.ngram_size - 1, 0) if self.ple_layer_ids else 0


class Qwen4ExpConfig(Qwen3_5Config):
    """Top-level Qwen4-Exp configuration with text/vision sub-configs."""

    model_type = "qwen4_exp"
    sub_configs = {
        "vision_config": Qwen4ExpVisionConfig,
        "text_config": Qwen4ExpTextConfig,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id: int = 248056,
        video_token_id: int = 248057,
        vision_start_token_id: int = 248053,
        vision_end_token_id: int = 248054,
        tie_word_embeddings: bool = False,
        rope_parameters: dict | None = None,
        **kwargs,
    ) -> None:
        if text_config is not None:
            kwargs.pop("split_ngram_parts", None)
        # Text-only exports may serialize text fields at the top level.
        text_kwargs = (
            dict(kwargs)
            if text_config is None
            and "hidden_size" in kwargs
            and "num_hidden_layers" in kwargs
            else None
        )
        if text_kwargs is not None:
            # The outer model type is not the decoder's model type.
            text_kwargs.pop("model_type", None)
            text_kwargs.setdefault("tie_word_embeddings", tie_word_embeddings)
            if rope_parameters is not None:
                text_kwargs.setdefault("rope_parameters", rope_parameters)
            text_config = text_kwargs
        super().__init__(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.rope_parameters = rope_parameters or getattr(
            self.text_config, "rope_parameters", {}
        )


__all__ = [
    "Qwen4ExpConfig",
    "Qwen4ExpTextConfig",
    "Qwen4ExpVisionConfig",
]
