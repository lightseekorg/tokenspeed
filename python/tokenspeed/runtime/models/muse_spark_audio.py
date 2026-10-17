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

"""Muse Spark audio tower: 1,920 mono 24 kHz samples per decoder-width token."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem

AUDIO_SAMPLES_PER_TOKEN = 1_920
AUDIO_INPUT_CHANNELS = 1


class MuseSparkAudioPatchEmbedder(nn.Module):
    """Fold ``patch`` consecutive steps into one, then project and normalize."""

    def __init__(
        self, patch: int, input_size: int, output_size: int, eps: float
    ) -> None:
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(input_size * patch, output_size, bias=True)
        self.norm = nn.RMSNorm(output_size, eps=eps, elementwise_affine=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, channels = value.shape
        folded = value.reshape(batch, length // self.patch, self.patch * channels)
        return self.norm(self.proj(folded))


class MuseSparkAudioCausalConv(nn.Conv1d):
    """Depthwise convolution that reads only the current and earlier steps."""

    def __init__(self, hidden_size: int, kernel_size: int) -> None:
        super().__init__(
            hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=True
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        padded = F.pad(value.transpose(1, 2), (self.kernel_size[0] - 1, 0))
        return super().forward(padded).transpose(1, 2)


class MuseSparkAudioMLP(nn.Module):
    """Two-layer GELU projection widened by the stage's MLP ratio."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(value)))


class MuseSparkAudioConvBlock(nn.Module):
    """Causal convolution followed by a GELU MLP, added to the residual."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        kernel_size: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.conv = MuseSparkAudioCausalConv(hidden_size, kernel_size)
        self.norm = nn.RMSNorm(hidden_size, eps=eps, elementwise_affine=True)
        self.mlp = MuseSparkAudioMLP(hidden_size, intermediate_size)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.conv(value)
        return self.mlp(self.norm(value)) + residual


class MuseSparkAudioLayer(nn.Module):
    """One patch embedder followed by ``depth`` convolution blocks."""

    def __init__(
        self,
        patch: int,
        input_size: int,
        hidden_size: int,
        intermediate_size: int,
        depth: int,
        kernel_size: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.patch_embedder = MuseSparkAudioPatchEmbedder(
            patch, input_size, hidden_size, eps
        )
        self.conv_blocks = nn.ModuleList(
            MuseSparkAudioConvBlock(hidden_size, intermediate_size, kernel_size, eps)
            for _ in range(depth)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.patch_embedder(value)
        for block in self.conv_blocks:
            value = block(value)
        return value


class MuseSparkAudioTower(nn.Module):
    """Encode mono waveforms into decoder-width tokens; ``patch_sizes`` multiply to the samples per token."""

    def __init__(
        self,
        out_hidden_size: int,
        patch_sizes: tuple[int, ...],
        hidden_sizes: tuple[int, ...],
        depths: tuple[int, ...],
        mlp_ratio: float,
        conv_kernel_size: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        if not len(patch_sizes) == len(hidden_sizes) == len(depths):
            raise ValueError("audio layers need one width and depth per patch size")
        if math.prod(patch_sizes) != AUDIO_SAMPLES_PER_TOKEN:
            raise ValueError("audio patch sizes must multiply to the samples per token")
        input_size = AUDIO_INPUT_CHANNELS
        layers = []
        for patch, width, depth in zip(patch_sizes, hidden_sizes, depths):
            layers.append(
                MuseSparkAudioLayer(
                    patch=patch,
                    input_size=input_size,
                    hidden_size=width,
                    intermediate_size=int(width * mlp_ratio),
                    depth=depth,
                    kernel_size=conv_kernel_size,
                    eps=rms_norm_eps,
                )
            )
            input_size = width
        self.layers = nn.ModuleList(layers)
        self.norm = nn.RMSNorm(input_size, eps=rms_norm_eps, elementwise_affine=True)
        self.proj = nn.Linear(input_size, out_hidden_size, bias=True)
        self.post_proj_norm = nn.RMSNorm(
            out_hidden_size, eps=rms_norm_eps, elementwise_affine=True
        )

    @classmethod
    def from_config(cls, raw: dict, hidden_size: int) -> MuseSparkAudioTower:
        """Build the audio tower from the checkpoint's ``audio_config``."""
        if raw["hidden_act"] != "gelu":
            raise ValueError("Muse Spark audio blocks activate through GELU")
        if int(raw["out_hidden_size"]) != hidden_size:
            raise ValueError("the audio tower must emit decoder-width tokens")
        if int(raw["num_hidden_layers"]) != len(raw["depths"]):
            raise ValueError("audio depths must cover every layer")
        return cls(
            out_hidden_size=hidden_size,
            patch_sizes=tuple(int(value) for value in raw["patch_sizes"]),
            hidden_sizes=tuple(int(value) for value in raw["hidden_sizes"]),
            depths=tuple(int(value) for value in raw["depths"]),
            mlp_ratio=float(raw["mlp_ratio"]),
            conv_kernel_size=int(raw["conv_kernel_size"]),
            rms_norm_eps=float(raw["rms_norm_eps"]),
        )

    def forward(self, waveforms: list[torch.Tensor]) -> torch.Tensor:
        """Encode each waveform, its tail zero-padded to a whole token, and concatenate the tokens."""
        outputs = []
        for waveform in waveforms:
            value = waveform.reshape(-1, AUDIO_INPUT_CHANNELS)
            padding = (-value.shape[0]) % AUDIO_SAMPLES_PER_TOKEN
            value = F.pad(value, (0, 0, 0, padding)).unsqueeze(0)
            for layer in self.layers:
                value = layer(value)
            value = self.post_proj_norm(self.proj(self.norm(value)))
            outputs.append(value.squeeze(0))
        return torch.cat(outputs, dim=0)

    def make_warmup_items(self) -> list[MultimodalDataItem]:
        """Return one silent single-token audio item for startup kernel compilation."""
        feature = self.proj.weight.new_zeros((AUDIO_SAMPLES_PER_TOKEN,))
        return [MultimodalDataItem(modality=Modality.AUDIO, feature=feature)]


__all__ = [
    "AUDIO_SAMPLES_PER_TOKEN",
    "MuseSparkAudioTower",
]
