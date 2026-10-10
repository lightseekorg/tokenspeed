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

"""Nemotron-H Omni: the Nemotron-H language model with C-RADIO vision and
Parakeet audio encoders (text output).

Each multimodal item fills one placeholder row per embedding, in order:

* image: ``feature`` holds ``[h * w, 3 * p * p]`` normalized patches (each
  patch flattened channel, row, column; patches row-major) and
  ``image_grid_hw`` the even patch grid ``(h, w)``; it embeds ``h * w / 4``
  rows (2x2 pixel shuffle);
* video: ``feature`` holds ``[frames * h * w, 3 * p * p]`` patches of
  equal-size frames and ``video_grid_thw`` is ``(frames, h, w)``; consecutive
  frames pair into tubelets (an odd last frame repeats), each embedding
  ``h * w / 4`` rows in its own placeholder run;
* audio: ``feature`` holds ``[frames, mel bins]`` log-mel features of
  consecutive clips and ``audio_clip_frames`` each clip's frame count; a clip
  embeds one row per 8 frames after subsampling.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
from tokenspeed_kernel.ops.activation import relu2
from torch import nn

from tokenspeed.runtime.configs.nemotron_omni_config import NemotronHOmniConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.nemotron_h import NemotronHForCausalLM
from tokenspeed.runtime.models.parakeet import ParakeetEncoder
from tokenspeed.runtime.models.radio import RadioVisionModel
from tokenspeed.runtime.multimodal.embedder import (
    EncoderSpec,
    MultimodalEmbedder,
    pad_input_tokens,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)

_RADIO_PREFIX = "vision_model.radio_model.model."
_PARAKEET_PREFIX = "sound_encoder.encoder."
# The vision projector's checkpoint indices into its nn.Sequential.
_MLP1_NAMES = {"0": "norm", "1": "linear1", "3": "linear2"}


class NemotronOmniProjector(nn.Module):
    """RMSNorm, linear, squared ReLU, linear: an encoder's map into the language model."""

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, bias: bool
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(input_size, eps=1e-5)
        self.linear1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear2 = nn.Linear(hidden_size, output_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.linear1(self.norm(x))
        return self.linear2(relu2(hidden, hidden, fp8_scale=None))

    def load_weight(self, name: str, loaded: torch.Tensor) -> None:
        param = dict(self.named_parameters())[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded)


def pixel_shuffle(features: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Merge each 2x2 patch block into one vector, ordered (block row, block column, channel)."""
    x = features.view(rows // 2, 2, cols // 2, 2, -1).permute(0, 2, 1, 3, 4)
    return x.reshape(rows * cols // 4, -1)


class NemotronH_Nano_Omni_Reasoning_V3(NemotronHForCausalLM):  # noqa: N801
    """Nemotron-H Omni for text generation; the class name is the HF architecture."""

    # The checkpoint's per-layer quantization names the language model under this prefix.
    quant_module_name_replacements = (("language_model.", ""),)

    def __init__(
        self,
        config: NemotronHOmniConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        *,
        is_multimodal_active: bool,
        mm_attention_backend: str | None,
    ) -> None:
        super().__init__(
            config=config.text_config, mapping=mapping, quant_config=quant_config
        )
        self.omni_config = config
        self.is_multimodal_active = is_multimodal_active
        self.visual: RadioVisionModel | None = None
        self.vision_projector: NemotronOmniProjector | None = None
        self.audio_tower: ParakeetEncoder | None = None
        self.audio_projector: NemotronOmniProjector | None = None
        self.multimodal_embedder: MultimodalEmbedder | None = None
        if not is_multimodal_active:
            return

        vision = config.vision_config
        text_hidden = config.text_config.hidden_size
        self.visual = RadioVisionModel(
            vision,
            mapping,
            final_norm=vision.final_norm,
            mm_attention_backend=mm_attention_backend,
            prefix="visual",
        )
        merge = round(1 / config.downsample_ratio)
        if merge != 2:
            raise NotImplementedError(f"Nemotron-H Omni {merge}x{merge} pixel shuffle")
        self.vision_projector = NemotronOmniProjector(
            vision.hidden_size * merge * merge,
            config.projector_hidden_size,
            text_hidden,
            bias=False,
        )
        sound = config.sound_config
        if sound is not None:
            self.audio_tower = ParakeetEncoder(sound)
            self.audio_projector = NemotronOmniProjector(
                sound.hidden_size,
                sound.projection_hidden_size,
                text_hidden,
                bias=sound.projection_bias,
            )
        self.multimodal_embedder = MultimodalEmbedder(encoder_mapping=mapping.vision)

    def pad_input_ids(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        return pad_input_tokens(input_ids, mm_inputs)

    def _project_vision(
        self, features: torch.Tensor, grids: list[tuple[int, int]]
    ) -> torch.Tensor:
        shuffled = []
        start = 0
        for rows, cols in grids:
            shuffled.append(
                pixel_shuffle(features[start : start + rows * cols], rows, cols)
            )
            start += rows * cols
        return self.vision_projector(torch.cat(shuffled))

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        device = self.visual.embedder.weight.device
        grids = [tuple(item.image_grid_hw.reshape(-1).tolist()) for item in items]
        patches = torch.cat([item.feature.to(device) for item in items])
        return self._project_vision(self.visual.embed_images(patches, grids), grids)

    def get_video_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        device = self.visual.embedder.weight.device
        tubelet = self.visual.temporal_patch_size
        embeddings = []
        for item in items:
            frames, rows, cols = item.video_grid_thw.reshape(-1).tolist()
            features = self.visual.embed_video(
                item.feature.to(device), frames, rows, cols
            )
            tubelets = -(-frames // tubelet)
            embeddings.append(self._project_vision(features, [(rows, cols)] * tubelets))
        return torch.cat(embeddings)

    def get_audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        device = self.audio_projector.linear1.weight.device
        clips = []
        for item in items:
            mel = item.feature.to(device)
            clips.extend(mel.split(item.audio_clip_frames.reshape(-1).tolist()))
        encoded = self.audio_tower.encode_clips(clips)
        return self.audio_projector(torch.cat(encoded))

    def make_image_warmup_items(self) -> list[MultimodalDataItem]:
        side = 32
        return [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=torch.zeros(side * side, self.visual.embedder.in_features),
                model_specific_data={"image_grid_hw": torch.tensor([side, side])},
            )
        ]

    def make_video_warmup_items(self) -> list[MultimodalDataItem]:
        side, frames = 32, 2
        return [
            MultimodalDataItem(
                modality=Modality.VIDEO,
                feature=torch.zeros(
                    frames * side * side, self.visual.embedder.in_features
                ),
                model_specific_data={
                    "video_grid_thw": torch.tensor([frames, side, side])
                },
            )
        ]

    def make_audio_warmup_items(self) -> list[MultimodalDataItem]:
        # One 30-second clip at 100 mel frames per second.
        frames = 3000
        mels = self.omni_config.sound_config.num_mel_bins
        return [
            MultimodalDataItem(
                modality=Modality.AUDIO,
                feature=torch.zeros(frames, mels),
                model_specific_data={"audio_clip_frames": torch.tensor([frames])},
            )
        ]

    def get_multimodal_encoder_specs(self) -> dict[Modality, EncoderSpec]:
        if self.visual is None:
            return {}
        specs = {
            Modality.IMAGE: EncoderSpec(
                self.get_image_feature, make_warmup_items=self.make_image_warmup_items
            ),
            Modality.VIDEO: EncoderSpec(
                self.get_video_feature, make_warmup_items=self.make_video_warmup_items
            ),
        }
        if self.audio_tower is not None:
            specs[Modality.AUDIO] = EncoderSpec(
                self.get_audio_feature, make_warmup_items=self.make_audio_warmup_items
            )
        return specs

    @torch.no_grad()
    def multimodal_input_embeds(
        self, input_ids: torch.Tensor, ctx: ForwardContext, multimodal_context
    ) -> torch.Tensor | None:
        if (
            multimodal_context is None
            or not multimodal_context.has_extend_inputs()
            or ctx.forward_mode.is_decode_or_idle()
        ):
            return None
        if self.multimodal_embedder is None:
            raise RuntimeError(
                "Nemotron-H Omni received multimodal inputs with its encoders disabled"
            )
        input_embeds, model_kwargs = self.multimodal_embedder.apply(
            input_ids=input_ids,
            text_embedding=self.model.embed_tokens,
            ctx=multimodal_context,
            encoders=self.get_multimodal_encoder_specs(),
            multimodal_model=self,
        )
        assert not model_kwargs, "Nemotron-H Omni embeds media as input rows only"
        return input_embeds

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        multimodal_context = kwargs.pop("multimodal_context", None)
        input_embeds = self.multimodal_input_embeds(input_ids, ctx, multimodal_context)
        if input_embeds is not None:
            kwargs["input_embeds"] = input_embeds
        return super().forward(ctx, input_ids, positions, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        super().load_weights(self._route_encoder_weights(weights))

    def _route_encoder_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Load the encoder and projector tensors; yield the language model's."""
        for name, loaded in weights:
            if name.startswith("language_model."):
                yield name.removeprefix("language_model."), loaded
            elif self.visual is None:
                continue
            elif name.startswith(_RADIO_PREFIX):
                self.visual.load_weight(name.removeprefix(_RADIO_PREFIX), loaded)
            elif name.startswith("mlp1."):
                index, suffix = name.removeprefix("mlp1.").split(".", 1)
                self.vision_projector.load_weight(
                    f"{_MLP1_NAMES[index]}.{suffix}", loaded
                )
            elif name.startswith(_PARAKEET_PREFIX):
                self.audio_tower.load_weight(
                    name.removeprefix(_PARAKEET_PREFIX), loaded
                )
            elif name.startswith("sound_projection."):
                self.audio_projector.load_weight(
                    name.removeprefix("sound_projection."), loaded
                )
            else:
                raise KeyError(f"Nemotron-H Omni has no tensor {name!r}")


EntryClass = [NemotronH_Nano_Omni_Reasoning_V3]
