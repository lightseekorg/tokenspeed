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

"""Nemotron-H Omni configuration: a Nemotron-H language model with a C-RADIO
vision encoder and a Parakeet audio encoder.

The checkpoint ``config.json`` (``model_type: NemotronH_Nano_Omni_Reasoning_V3``)
loads unmodified: ``llm_config`` becomes TokenSpeed's ``NemotronHConfig`` as
``text_config``, and the text-model attributes generic engine code reads are
forwarded to it.
"""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig

from tokenspeed.runtime.configs.nemotron_h_config import NemotronHConfig

# timm ViT name -> (hidden size, layers, heads, MLP size).
_VIT_DIMS = {
    "vit_base_patch16_224": (768, 12, 12, 3072),
    "vit_large_patch16_224": (1024, 24, 16, 4096),
    "vit_huge_patch16_224": (1280, 32, 16, 5120),
}


class RadioVisionConfig(PretrainedConfig):
    """C-RADIO ViT geometry, read from the checkpoint's RADIO ``args``."""

    model_type = "radio_vision"

    def __init__(
        self,
        *,
        args: dict[str, Any] | None = None,
        patch_size: int = 16,
        video_temporal_patch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        self.args = dict(args or {})
        self.patch_size = patch_size
        self.video_temporal_patch_size = video_temporal_patch_size
        super().__init__(**kwargs)

    @property
    def vit_dims(self) -> tuple[int, int, int, int]:
        name = self.args["model"]
        if name not in _VIT_DIMS:
            raise NotImplementedError(f"C-RADIO backbone {name!r}")
        return _VIT_DIMS[name]

    @property
    def hidden_size(self) -> int:
        return self.vit_dims[0]

    @property
    def num_hidden_layers(self) -> int:
        return self.vit_dims[1]

    @property
    def num_attention_heads(self) -> int:
        return self.vit_dims[2]

    @property
    def intermediate_size(self) -> int:
        return self.vit_dims[3]

    @property
    def position_grid_size(self) -> int:
        """Side of the square position-embedding grid the CPE interpolates from."""
        return self.args["cpe_max_size"] // self.patch_size

    @property
    def num_prefix_tokens(self) -> int:
        """Class tokens (one per distinct teacher) plus registers, prepended to every image."""
        if self.args["cls_token_per_teacher"]:
            num_cls = len({teacher["name"] for teacher in self.args["teachers"]})
        else:
            num_cls = 1
        multiple = self.args["register_multiple"]
        # C-RADIO pads to the next multiple, adding a full multiple when already aligned.
        return num_cls + (multiple - num_cls % multiple if multiple else 0)

    @property
    def final_norm(self) -> bool:
        return self.args["model_norm"]


class ParakeetAudioConfig(PretrainedConfig):
    """Parakeet (FastConformer) encoder and its projector into the language model."""

    model_type = "parakeet_audio"

    def __init__(
        self,
        *,
        hidden_size: int = 1024,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 24,
        intermediate_size: int = 4096,
        attention_bias: bool = False,
        conv_kernel_size: int = 9,
        convolution_bias: bool = False,
        subsampling_conv_channels: int = 256,
        subsampling_conv_kernel_size: int = 3,
        subsampling_conv_stride: int = 2,
        subsampling_factor: int = 8,
        num_mel_bins: int = 128,
        projection_hidden_size: int = 4096,
        projection_bias: bool = False,
        sampling_rate: int = 16000,
        **kwargs: Any,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.attention_bias = attention_bias
        self.conv_kernel_size = conv_kernel_size
        self.convolution_bias = convolution_bias
        self.subsampling_conv_channels = subsampling_conv_channels
        self.subsampling_conv_kernel_size = subsampling_conv_kernel_size
        self.subsampling_conv_stride = subsampling_conv_stride
        self.subsampling_factor = subsampling_factor
        self.num_mel_bins = num_mel_bins
        self.projection_hidden_size = projection_hidden_size
        self.projection_bias = projection_bias
        self.sampling_rate = sampling_rate
        super().__init__(**kwargs)


class NemotronHOmniConfig(PretrainedConfig):
    """Container configuration for Nemotron-H Omni checkpoints."""

    model_type = "NemotronH_Nano_Omni_Reasoning_V3"

    def __init__(
        self,
        *,
        llm_config: dict[str, Any] | NemotronHConfig | None = None,
        vision_config: dict[str, Any] | RadioVisionConfig | None = None,
        sound_config: dict[str, Any] | ParakeetAudioConfig | None = None,
        img_context_token_id: int = 18,
        sound_context_token_id: int | None = None,
        downsample_ratio: float = 0.5,
        ps_version: str = "v2",
        vit_hidden_size: int = 1280,
        projector_hidden_size: int = 20480,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        if ps_version != "v2":
            raise NotImplementedError(f"Nemotron-H Omni pixel shuffle {ps_version!r}")
        self.text_config = (
            llm_config
            if isinstance(llm_config, NemotronHConfig)
            else NemotronHConfig(**(llm_config or {}))
        )
        self.vision_config = (
            vision_config
            if isinstance(vision_config, RadioVisionConfig)
            else RadioVisionConfig(**(vision_config or {}))
        )
        self.sound_config = (
            sound_config
            if sound_config is None or isinstance(sound_config, ParakeetAudioConfig)
            else ParakeetAudioConfig(**sound_config)
        )
        self.image_token_id = img_context_token_id
        # Video tubelets embed on the image placeholder; the checkpoint's video id is unused.
        self.video_token_id = img_context_token_id
        self.audio_token_id = sound_context_token_id
        self.downsample_ratio = downsample_ratio
        self.ps_version = ps_version
        self.vit_hidden_size = vit_hidden_size
        self.projector_hidden_size = projector_hidden_size
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_text_config(self, *args: Any, **kwargs: Any) -> NemotronHConfig:
        return self.text_config

    @property
    def cache_layer_types(self) -> list[str]:
        return self.text_config.cache_layer_types

    @property
    def sliding_window(self) -> int | None:
        return self.text_config.sliding_window

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
    def num_attention_heads(self) -> int:
        return self.text_config.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.text_config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return self.text_config.head_dim
