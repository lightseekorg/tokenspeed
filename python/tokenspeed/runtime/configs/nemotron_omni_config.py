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

The checkpoint ``config.json`` (``model_type`` ``NemotronH_Nano_Omni_Reasoning_V3``
or ``nemotron_h_omni``) loads unmodified: ``llm_config`` becomes TokenSpeed's
``NemotronHConfig`` as ``text_config``, and the text-model attributes generic
engine code reads are forwarded to it.
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

# RADIO ``args`` layer settings the encoder implements; a checkpoint may only repeat these values.
_RADIO_FIXED_ARGS = {
    "hidden_act": "gelu",
    "qkv_bias": True,
    "qk_normalization": False,
    "norm_type": "layer_norm",
}

# The ViT shape a transformers-format RADIO config states beside ``args``.
_RADIO_SHAPE_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "mlp_ratio",
    "hidden_act",
    "qkv_bias",
    "use_swiglu_ffn",
    "layerscale_value",
    "num_cls_tokens",
    "num_registers",
)


class RadioVisionConfig(PretrainedConfig):
    """C-RADIO ViT geometry, read from the checkpoint's RADIO ``args``."""

    model_type = "radio_vision"

    def __init__(
        self,
        *,
        args: dict[str, Any] | None = None,
        patch_size: int = 16,
        video_temporal_patch_size: int = 1,
        separate_video_embedder: bool = True,
        **kwargs: Any,
    ) -> None:
        self.args = dict(args or {})
        for name, value in _RADIO_FIXED_ARGS.items():
            if self.args.get(name, value) != value:
                raise ValueError(
                    f"C-RADIO args.{name}={self.args[name]!r} is not supported; "
                    f"the encoder implements {value!r}"
                )
        if "layer_norm_eps" in kwargs:
            eps = kwargs.pop("layer_norm_eps")
            if self.args.setdefault("layer_norm_eps", eps) != eps:
                raise ValueError(
                    f"C-RADIO layer_norm_eps={eps} disagrees with "
                    f"args.layer_norm_eps={self.args['layer_norm_eps']}"
                )
        stated = {
            name: kwargs.pop(name) for name in _RADIO_SHAPE_FIELDS if name in kwargs
        }
        self.patch_size = patch_size
        self.video_temporal_patch_size = video_temporal_patch_size
        self.separate_video_embedder = separate_video_embedder
        super().__init__(**kwargs)
        if stated:
            shape = self.shape_from_args()
            mismatched = {
                name: value for name, value in stated.items() if value != shape[name]
            }
            if mismatched:
                raise ValueError(
                    f"C-RADIO vision_config {mismatched} disagrees with the "
                    f"{self.args['model']} encoder, which has {shape}"
                )

    def shape_from_args(self) -> dict[str, Any]:
        """The ViT shape ``args`` describes, in the transformers RADIO config's fields."""
        hidden_size, num_layers, num_heads, mlp_size = self.vit_dims
        return {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
            "mlp_ratio": mlp_size / hidden_size,
            "hidden_act": "gelu",
            "qkv_bias": True,
            "use_swiglu_ffn": False,
            # LayerScale left at identity: checkpoints carry no scale tensors.
            "layerscale_value": 1.0,
            "num_cls_tokens": self.num_cls_tokens,
            "num_registers": self.num_registers,
        }

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
    def num_cls_tokens(self) -> int:
        """One class token per distinct teacher, or a single one."""
        if self.args["cls_token_per_teacher"]:
            return len({teacher["name"] for teacher in self.args["teachers"]})
        return 1

    @property
    def num_registers(self) -> int:
        multiple = self.args["register_multiple"]
        # C-RADIO pads to the next multiple, adding a full multiple when already aligned.
        return multiple - self.num_cls_tokens % multiple if multiple else 0

    @property
    def num_prefix_tokens(self) -> int:
        """Class tokens plus registers, prepended to every image."""
        return self.num_cls_tokens + self.num_registers

    @property
    def image_patch_budget(self) -> int:
        """The dynamic-resolution planner's per-image patch budget."""
        return self.args["max_num_patches"]

    @property
    def image_patch_floor(self) -> int:
        """The dynamic-resolution planner's per-image patch minimum."""
        return self.args["min_num_patches"]

    @property
    def final_norm(self) -> bool:
        # A transformers-format config has no ``model_norm``: its ViT ends without a norm.
        return self.args.get("model_norm", False)

    @property
    def layer_norm_eps(self) -> float:
        """LayerNorm epsilon of the blocks and the final norm; RADIO's 1e-6 unless ``args`` sets it."""
        return self.args.get("layer_norm_eps", 1e-6)


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
        projection_eps: float = 1e-5,
        sampling_rate: int = 16000,
        hidden_act: str = "silu",
        **kwargs: Any,
    ) -> None:
        if hidden_act != "silu":
            raise ValueError(
                f"Parakeet hidden_act {hidden_act!r} is not supported; the encoder implements silu"
            )
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
        self.projection_eps = projection_eps
        self.sampling_rate = sampling_rate
        self.hidden_act = hidden_act
        super().__init__(**kwargs)


# The sound_config fields that shape the Parakeet encoder and its projector. The ref
# engine requires the projector's and fills the encoder's from transformers' defaults,
# which differ from this checkpoint family's, so a checkpoint must state every one.
_PARAKEET_ARCHITECTURE_FIELDS = (
    "hidden_size",
    "num_attention_heads",
    "num_hidden_layers",
    "intermediate_size",
    "conv_kernel_size",
    "convolution_bias",
    "subsampling_conv_channels",
    "subsampling_conv_kernel_size",
    "subsampling_conv_stride",
    "subsampling_factor",
    "num_mel_bins",
    "projection_hidden_size",
    "projection_bias",
    "sampling_rate",
)


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
        if isinstance(sound_config, dict):
            missing = [
                f for f in _PARAKEET_ARCHITECTURE_FIELDS if f not in sound_config
            ]
            if missing:
                raise ValueError(f"Nemotron-H Omni sound_config lacks {missing}")
            sound_config = ParakeetAudioConfig(**sound_config)
        self.sound_config = sound_config
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

    def to_dict(self) -> dict[str, Any]:
        """The checkpoint's schema, which ``__init__`` reads back."""
        output = super().to_dict()
        output["llm_config"] = output.pop("text_config")
        output["img_context_token_id"] = output.pop("image_token_id")
        output["sound_context_token_id"] = output.pop("audio_token_id")
        del output["video_token_id"]
        return output

    @property
    def vision_final_layernorm(self) -> bool:
        """Whether a LayerNorm runs on the ViT output ahead of the projector.

        Checkpoints whose language model has MTP layers carry it: their trainer
        builds every block stack, the vision tower's too, with a final LayerNorm.
        """
        return self.text_config.num_nextn_predict_layers > 0

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
