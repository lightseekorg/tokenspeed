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

"""Shared Qwen3-ASR/Qwen3-Omni audio encoder.

The gateway extracts log-Mel features with the Whisper-compatible Qwen3
frontend. Each multimodal item contains one ``[num_mel_bins, num_frames]``
tensor (a leading singleton batch dimension is also accepted) and optionally
carries ``audio_feature_lengths`` or ``feature_attention_mask`` in
``model_specific_data``. This module packs those request-local tensors and
implements the common Qwen3 audio tower used by both Qwen3-ASR and the
Qwen3-Omni thinker.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Integral
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.attention.mm_encoder_attention import (
    MultimodalEncoderAttention,
)
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.multimodal.inputs import MultimodalDataItem
from tokenspeed.runtime.utils import add_prefix


def _cnn_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """Lengths after the tower's three stride-2, padding-1 convolutions."""
    lengths = input_lengths
    for _ in range(3):
        lengths = (lengths - 1) // 2 + 1
    return lengths


def qwen3_audio_output_lengths(
    input_lengths: torch.Tensor | Sequence[int] | int,
    *,
    n_window: int = 50,
) -> torch.Tensor:
    """Return language-token counts produced for log-Mel frame lengths.

    Qwen processes audio in chunks of ``2 * n_window`` frames.  Convolution
    padding is applied independently to every chunk, so applying a single
    ``ceil(length / 8)`` to a long clip would under-count tokens at chunk
    boundaries.

    Args:
        input_lengths: Scalar or one-dimensional Mel-frame lengths.
        n_window: Half of the frontend chunk size from the audio config.

    Returns:
        A one-dimensional ``torch.long`` tensor with one token count per input.
    """
    if n_window <= 0:
        raise ValueError(f"n_window must be positive, got {n_window}")
    if isinstance(input_lengths, Integral):
        lengths = torch.tensor([int(input_lengths)], dtype=torch.long)
    elif isinstance(input_lengths, torch.Tensor):
        lengths = input_lengths.to(dtype=torch.long)
        if lengths.ndim == 0:
            lengths = lengths.unsqueeze(0)
    else:
        lengths = torch.as_tensor(list(input_lengths), dtype=torch.long)
    if lengths.ndim != 1:
        raise ValueError(
            f"audio feature lengths must be one-dimensional, got {lengths.shape}"
        )
    if torch.any(lengths < 0):
        raise ValueError("audio feature lengths cannot be negative")

    chunk_size = 2 * n_window
    full_chunks = lengths // chunk_size
    tail = lengths % chunk_size
    full_chunk_output = int(
        _cnn_output_lengths(torch.tensor([chunk_size], dtype=torch.long)).item()
    )
    tail_output = _cnn_output_lengths(tail)
    tail_output = torch.where(tail == 0, torch.zeros_like(tail), tail_output)
    return full_chunks * full_chunk_output + tail_output


def _one_item_feature_length(item: MultimodalDataItem, max_frames: int) -> int:
    explicit_length = item.model_specific_data.get("audio_feature_lengths")
    if explicit_length is not None:
        if isinstance(explicit_length, torch.Tensor):
            if explicit_length.numel() != 1:
                raise ValueError(
                    "one audio item must carry exactly one audio_feature_lengths "
                    f"value, got shape {explicit_length.shape}"
                )
            length = int(explicit_length.reshape(-1)[0].item())
        elif isinstance(explicit_length, Integral):
            length = int(explicit_length)
        else:
            raise TypeError(
                "audio_feature_lengths must be an integer or tensor, got "
                f"{type(explicit_length).__name__}"
            )
    else:
        attention_mask = item.model_specific_data.get("feature_attention_mask")
        if attention_mask is None:
            length = max_frames
        else:
            attention_mask = torch.as_tensor(attention_mask)
            if attention_mask.ndim == 2 and attention_mask.shape[0] == 1:
                attention_mask = attention_mask.squeeze(0)
            if attention_mask.ndim != 1:
                raise ValueError(
                    "one audio item must carry a one-dimensional "
                    f"feature_attention_mask, got {attention_mask.shape}"
                )
            if attention_mask.numel() != max_frames:
                raise ValueError(
                    "feature_attention_mask length does not match the audio "
                    f"feature: {attention_mask.numel()} != {max_frames}"
                )
            # Whisper masks are right-padded.  Requiring a contiguous prefix
            # prevents silently packing frames from a malformed sparse mask.
            mask = attention_mask.to(dtype=torch.bool)
            length = int(mask.sum().item())
            expected = torch.arange(max_frames, device=mask.device) < length
            if not torch.equal(mask, expected):
                raise ValueError("feature_attention_mask must be right-padded")

    if length <= 0 or length > max_frames:
        raise ValueError(
            f"audio feature length must be in [1, {max_frames}], got {length}"
        )
    return length


def pack_qwen3_audio_features(
    items: Sequence[MultimodalDataItem],
    *,
    num_mel_bins: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack itemized log-Mel tensors for :class:`Qwen3AudioEncoder`.

    Returns a feature tensor shaped ``[num_mel_bins, sum(valid_frames)]`` and
    a length tensor shaped ``[num_items]``.  Padding is removed before packing.
    """
    if not items:
        raise ValueError("at least one audio item is required")

    features: list[torch.Tensor] = []
    lengths: list[int] = []
    for item in items:
        feature = item.feature
        if not isinstance(feature, torch.Tensor):
            raise TypeError(
                "audio item feature must be materialized as a torch.Tensor, got "
                f"{type(feature).__name__}"
            )
        if feature.ndim == 3 and feature.shape[0] == 1:
            feature = feature.squeeze(0)
        if feature.ndim != 2:
            raise ValueError(
                "audio feature must have shape [mel, frames] or [1, mel, frames], "
                f"got {feature.shape}"
            )
        if feature.shape[0] != num_mel_bins:
            raise ValueError(
                f"expected {num_mel_bins} Mel bins, got feature shape {feature.shape}"
            )
        length = _one_item_feature_length(item, feature.shape[1])
        features.append(feature[:, :length].to(device=device, dtype=dtype))
        lengths.append(length)

    return (
        torch.cat(features, dim=1).contiguous(),
        torch.tensor(lengths, dtype=torch.long, device=device),
    )


class SinusoidsPositionEmbedding(nn.Module):
    """Non-trainable absolute position embedding used by the audio tower."""

    def __init__(self, length: int, channels: int, max_timescale: int = 10000):
        super().__init__()
        if channels % 2:
            raise ValueError("sinusoidal position embeddings need even channels")
        log_increment = torch.log(torch.tensor(float(max_timescale))) / (
            channels // 2 - 1
        )
        inv_timescales = torch.exp(-log_increment * torch.arange(channels // 2))
        positions = torch.arange(length).unsqueeze(1) * inv_timescales.unsqueeze(0)
        self.register_buffer(
            "positional_embedding",
            torch.cat([positions.sin(), positions.cos()], dim=1),
            persistent=False,
        )

    def forward(self, sequence_length: int) -> torch.Tensor:
        return self.positional_embedding[:sequence_length]


class Qwen3AudioEncoderLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = MultimodalEncoderAttention(
            embed_dim=config.d_model,
            num_heads=config.encoder_attention_heads,
            mapping=mapping,
            qkv_bias=True,
            proj_bias=True,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            mm_attention_backend=mm_attention_backend,
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.activation_fn = ACT2FN[config.activation_function]

        vision = mapping.vision
        self.fc1 = ColumnParallelLinear(
            config.d_model,
            config.encoder_ffn_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
        )
        self.fc2 = RowParallelLinear(
            config.encoder_ffn_dim,
            config.d_model,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
            reduce_results=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        residual = hidden_states
        normed = self.self_attn_layer_norm(hidden_states)
        attended = self.self_attn(
            normed.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        ).squeeze(0)
        hidden_states = residual + attended

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            limit = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = hidden_states.clamp(min=-limit, max=limit)
        return hidden_states


class Qwen3AudioEncoder(nn.Module):
    """Parameterizable Qwen audio tower shared by ASR and Omni thinker."""

    @staticmethod
    def _normalize_attention_backend(name: str | None) -> str | None:
        # The cuDNN path consumes vision-specific sequence metadata. Audio can
        # still use the platform default while an Omni vision tower uses cuDNN.
        return None if name == "flashinfer_cudnn" else name

    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_mel_bins = int(config.num_mel_bins)
        self.n_window = int(config.n_window)
        self.n_window_infer = int(config.n_window_infer)
        self.conv_chunksize = int(config.conv_chunksize)
        mm_attention_backend = self._normalize_attention_backend(mm_attention_backend)

        self.positional_embedding = SinusoidsPositionEmbedding(
            int(config.max_source_positions), int(config.d_model)
        )
        self.conv2d1 = nn.Conv2d(
            1, config.downsample_hidden_size, kernel_size=3, stride=2, padding=1
        )
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        conv_out_dim = config.downsample_hidden_size * (
            (((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        )
        self.conv_out = ReplicatedLinear(
            conv_out_dim,
            config.d_model,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("conv_out", prefix),
        )
        self.layers = nn.ModuleList(
            [
                Qwen3AudioEncoderLayer(
                    config,
                    mapping,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{index}", prefix),
                    mm_attention_backend=mm_attention_backend,
                )
                for index in range(config.encoder_layers)
            ]
        )
        self.ln_post = nn.LayerNorm(config.d_model)
        self.proj1 = ReplicatedLinear(
            config.d_model,
            config.d_model,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("proj1", prefix),
        )
        self.act = ACT2FN[config.activation_function]
        self.proj2 = ReplicatedLinear(
            config.d_model,
            config.output_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("proj2", prefix),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.conv2d1.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.conv2d1.weight.device

    def encode(self, items: Sequence[MultimodalDataItem]) -> torch.Tensor:
        input_features, feature_lengths = pack_qwen3_audio_features(
            items,
            num_mel_bins=self.num_mel_bins,
            device=self.device,
            dtype=self.dtype,
        )
        return self(input_features, feature_lengths)

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Encode packed features into concatenated LM-space audio tokens.

        Args:
            input_features: Packed log-Mel features shaped
                ``[num_mel_bins, sum(feature_lengths)]``.
            feature_lengths: Valid frame count for every original audio item.

        Returns:
            Audio embeddings shaped ``[sum(output_lengths), output_dim]`` in
            the same item order as ``feature_lengths``.
        """
        input_features = input_features.to(device=self.device, dtype=self.dtype)
        feature_lengths = feature_lengths.to(device=self.device, dtype=torch.long)
        if input_features.ndim != 2 or input_features.shape[0] != self.num_mel_bins:
            raise ValueError(
                f"expected packed features [{self.num_mel_bins}, frames], got "
                f"{input_features.shape}"
            )
        if feature_lengths.ndim != 1 or feature_lengths.numel() == 0:
            raise ValueError("feature_lengths must be a non-empty vector")
        if torch.any(feature_lengths <= 0):
            raise ValueError("feature_lengths must be positive")
        if int(feature_lengths.sum().item()) != input_features.shape[1]:
            raise ValueError(
                "packed feature width does not match feature_lengths: "
                f"{input_features.shape[1]} != {int(feature_lengths.sum().item())}"
            )

        chunk_size = self.n_window * 2
        chunk_lengths_list: list[int] = []
        for length in feature_lengths.tolist():
            full_chunks, tail = divmod(int(length), chunk_size)
            chunk_lengths_list.extend([chunk_size] * full_chunks)
            if tail:
                chunk_lengths_list.append(tail)
        chunk_lengths = torch.tensor(
            chunk_lengths_list, dtype=torch.long, device=self.device
        )

        chunks = input_features.transpose(0, 1).split(chunk_lengths_list, dim=0)
        padded_features = nn.utils.rnn.pad_sequence(chunks, batch_first=True).transpose(
            1, 2
        )
        chunk_output_lengths = _cnn_output_lengths(chunk_lengths)
        max_chunk_output = int(chunk_output_lengths.max().item())
        output_mask = torch.arange(max_chunk_output, device=self.device).unsqueeze(
            0
        ) < chunk_output_lengths.unsqueeze(1)

        padded_features = padded_features.unsqueeze(1)
        encoded_chunks: list[torch.Tensor] = []
        for conv_input in padded_features.split(self.conv_chunksize, dim=0):
            hidden = F.gelu(self.conv2d1(conv_input))
            hidden = F.gelu(self.conv2d2(hidden))
            hidden = F.gelu(self.conv2d3(hidden))
            encoded_chunks.append(hidden)
        hidden = torch.cat(encoded_chunks, dim=0)

        batch, channels, frequency, time = hidden.shape
        hidden, _ = self.conv_out(
            hidden.permute(0, 3, 1, 2)
            .contiguous()
            .view(batch, time, channels * frequency)
        )
        if hidden.shape[1] > self.positional_embedding.positional_embedding.shape[0]:
            raise ValueError(
                "audio chunk exceeds max_source_positions after convolution: "
                f"{hidden.shape[1]}"
            )
        hidden = hidden + self.positional_embedding(hidden.shape[1]).to(hidden.dtype)
        hidden_states = hidden[output_mask]

        output_lengths = qwen3_audio_output_lengths(
            feature_lengths, n_window=self.n_window
        ).to(device=self.device)
        chunks_per_attention_window = max(1, self.n_window_infer // chunk_size)
        attention_window = max_chunk_output * chunks_per_attention_window
        attention_lengths: list[int] = []
        for length in output_lengths.tolist():
            full_windows, tail = divmod(int(length), attention_window)
            attention_lengths.extend([attention_window] * full_windows)
            if tail:
                attention_lengths.append(tail)
        cu_seqlens = torch.tensor(
            [0, *attention_lengths], dtype=torch.int32, device=self.device
        ).cumsum(0, dtype=torch.int32)
        max_seqlen = max(attention_lengths)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens, max_seqlen)

        hidden_states = self.ln_post(hidden_states)
        hidden_states, _ = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.proj2(hidden_states)
        return hidden_states

    @staticmethod
    def map_weight_name(name: str) -> tuple[str, str | None]:
        """Map an HF audio-tower name to ``(TokenSpeed name, shard id)``."""
        marker = "audio_tower."
        if marker in name:
            name = name.split(marker, 1)[1]
        name = name.replace("self_attn.out_proj.", "self_attn.proj.")
        for checkpoint_name, shard_id in (
            ("q_proj", "q"),
            ("k_proj", "k"),
            ("v_proj", "v"),
        ):
            needle = f"self_attn.{checkpoint_name}."
            if needle in name:
                return name.replace(needle, "self_attn.qkv_proj."), shard_id
        return name, None

    def load_weight(self, name: str, loaded_weight: torch.Tensor) -> str | None:
        """Load one audio-tower checkpoint tensor.

        ``name`` may be relative to the tower or retain a
        ``thinker.audio_tower``/``audio_tower`` prefix.  Q/K/V checkpoint
        projections are fused into TokenSpeed's ``qkv_proj`` parameter and the
        HF ``out_proj`` name is mapped to ``MultimodalEncoderAttention.proj``.
        """
        name, shard_id = self.map_weight_name(name)
        params = dict(self.named_parameters(remove_duplicate=False))
        if name not in params:
            return None
        param = params[name]
        if shard_id is not None:
            param.weight_loader(param, loaded_weight, shard_id)
        else:
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, loaded_weight)
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        for name, tensor in weights:
            loaded_name = self.load_weight(name, tensor)
            if loaded_name is not None:
                loaded.add(loaded_name)
        return loaded


__all__ = [
    "Qwen3AudioEncoder",
    "Qwen3AudioEncoderLayer",
    "pack_qwen3_audio_features",
    "qwen3_audio_output_lengths",
]
