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

"""Parakeet (FastConformer) audio encoder over log-mel clips.

A stack of stride-2 convolutions subsamples the mel frames 8x, then every
conformer block applies a half-step feed-forward, self-attention with
Transformer-XL relative positions, a depthwise convolution module and a
second half-step feed-forward. Module names follow the checkpoint
(``sound_encoder.encoder.*``). Clips are encoded without padding: equal-length
clips share one batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tokenspeed.runtime.configs.nemotron_omni_config import ParakeetAudioConfig
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader


def _relative_positions(length: int, dim: int, like: torch.Tensor) -> torch.Tensor:
    """Sinusoids of the offsets ``length - 1`` down to ``1 - length``, sin and cos interleaved."""
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(0, dim, 2, device=like.device, dtype=torch.float32) / dim)
    )
    offsets = torch.arange(
        length - 1, -length, -1, device=like.device, dtype=torch.float32
    )
    freqs = offsets[:, None] * inv_freq[None, :]
    return torch.stack([freqs.sin(), freqs.cos()], dim=-1).flatten(1).to(like.dtype)


class ParakeetSubsampling(nn.Module):
    def __init__(self, config: ParakeetAudioConfig) -> None:
        super().__init__()
        kernel = config.subsampling_conv_kernel_size
        stride = config.subsampling_conv_stride
        channels = config.subsampling_conv_channels
        padding = (kernel - 1) // 2
        num_stages = config.subsampling_factor.bit_length() - 1
        layers: list[nn.Module] = [
            nn.Conv2d(1, channels, kernel, stride=stride, padding=padding),
            nn.ReLU(),
        ]
        for _ in range(num_stages - 1):
            layers += [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel,
                    stride=stride,
                    padding=padding,
                    groups=channels,
                ),
                nn.Conv2d(channels, channels, 1),
                nn.ReLU(),
            ]
        self.layers = nn.ModuleList(layers)
        mel_out = config.num_mel_bins // stride**num_stages
        self.linear = nn.Linear(channels * mel_out, config.hidden_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``[batch, frames, mels]`` -> ``[batch, frames / 8, hidden]``."""
        x = features.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        return self.linear(x.transpose(1, 2).flatten(2))


class ParakeetFeedForward(nn.Module):
    def __init__(self, config: ParakeetAudioConfig) -> None:
        super().__init__()
        bias = config.attention_bias
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size, bias)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.silu(self.linear1(x)))


class ParakeetAttention(nn.Module):
    """Self-attention with content and position biases over relative offsets."""

    def __init__(self, config: ParakeetAudioConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        bias = config.attention_bias
        self.num_heads = config.num_attention_heads
        self.head_dim = hidden // self.num_heads
        self.q_proj = nn.Linear(hidden, hidden, bias)
        self.k_proj = nn.Linear(hidden, hidden, bias)
        self.v_proj = nn.Linear(hidden, hidden, bias)
        self.o_proj = nn.Linear(hidden, hidden, bias)
        self.relative_k_proj = nn.Linear(hidden, hidden, bias=False)
        self.bias_u = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        self.bias_v = nn.Parameter(torch.empty(self.num_heads, self.head_dim))

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q, k, v = (
            proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        rel_k = self.relative_k_proj(positions).view(-1, self.num_heads, self.head_dim)
        # Position scores over the 2*length-1 offsets, shifted so column j is key j.
        scores = (q + self.bias_v[:, None]) @ rel_k.permute(1, 2, 0)
        scores = F.pad(scores, (1, 0)).view(batch, self.num_heads, -1, length)
        scores = scores[:, :, 1:].reshape(batch, self.num_heads, length, -1)
        position_bias = scores[..., :length] * self.head_dim**-0.5
        out = F.scaled_dot_product_attention(
            q + self.bias_u[:, None], k, v, attn_mask=position_bias
        )
        return self.o_proj(out.transpose(1, 2).flatten(2))


class ParakeetBatchNorm(nn.Module):
    """Inference batch norm over channels, from the checkpoint's running statistics."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels))
        self.bias = nn.Parameter(torch.empty(channels))
        self.running_mean = nn.Parameter(torch.empty(channels), requires_grad=False)
        self.running_var = nn.Parameter(torch.empty(channels), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=False,
            eps=1e-5,
        )


class ParakeetConvModule(nn.Module):
    def __init__(self, config: ParakeetAudioConfig) -> None:
        super().__init__()
        channels = config.hidden_size
        kernel = config.conv_kernel_size
        bias = config.convolution_bias
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, 1, bias=bias)
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel,
            padding=(kernel - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = ParakeetBatchNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.glu(self.pointwise_conv1(x.transpose(1, 2)), dim=1)
        x = F.silu(self.norm(self.depthwise_conv(x)))
        return self.pointwise_conv2(x).transpose(1, 2)


class ParakeetBlock(nn.Module):
    def __init__(self, config: ParakeetAudioConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.feed_forward1 = ParakeetFeedForward(config)
        self.self_attn = ParakeetAttention(config)
        self.conv = ParakeetConvModule(config)
        self.feed_forward2 = ParakeetFeedForward(config)
        self.norm_feed_forward1 = nn.LayerNorm(hidden)
        self.norm_self_att = nn.LayerNorm(hidden)
        self.norm_conv = nn.LayerNorm(hidden)
        self.norm_feed_forward2 = nn.LayerNorm(hidden)
        self.norm_out = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), positions)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class ParakeetEncoder(nn.Module):
    def __init__(self, config: ParakeetAudioConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.subsampling = ParakeetSubsampling(config)
        self.layers = nn.ModuleList(
            ParakeetBlock(config) for _ in range(config.num_hidden_layers)
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.subsampling.linear.weight.dtype

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Encode equal-length clips: ``[clips, frames, mels]`` -> ``[clips, frames / 8, hidden]``."""
        x = self.subsampling(features.to(self.dtype))
        positions = _relative_positions(x.shape[1], self.hidden_size, x)
        for layer in self.layers:
            x = layer(x, positions)
        return x

    def encode_clips(self, clips: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode ``[frames, mels]`` clips of any lengths, batching equal lengths."""
        by_length: dict[int, list[int]] = {}
        for index, clip in enumerate(clips):
            by_length.setdefault(clip.shape[0], []).append(index)
        outputs: list[torch.Tensor | None] = [None] * len(clips)
        for indices in by_length.values():
            encoded = self(torch.stack([clips[i] for i in indices]))
            for i, out in zip(indices, encoded):
                outputs[i] = out
        return outputs

    def load_weight(self, name: str, loaded: torch.Tensor) -> None:
        """Load one checkpoint tensor named relative to the encoder."""
        if name.endswith(".num_batches_tracked"):
            return
        param = dict(self.named_parameters())[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded)
