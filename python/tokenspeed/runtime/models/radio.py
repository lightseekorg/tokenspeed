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

"""C-RADIO vision encoder: a pre-norm ViT over packed images of any size.

Each image arrives as flattened ``(channel, row, column)`` patches in row-major
patch order. A linear embedding (or, for video, one over a tubelet of
consecutive frames) maps patches to the hidden size; the position embedding
is the learned square grid bilinearly resized to the image's longer side and
cropped to its shape (CPE); class and register tokens are prepended. Images
attend only within themselves. The output drops the prefix tokens and keeps
one vector per patch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import accumulate
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from tokenspeed.runtime.configs.nemotron_omni_config import RadioVisionConfig
from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.layers.attention.mm_encoder_attention import VisionAttention
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.multimodal.encoder_cudagraph import (
    PADDING_SEQUENCE_ROWS,
    pad_cu_seqlens,
)


class RadioMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class RadioBlock(nn.Module):
    def __init__(
        self,
        config: RadioVisionConfig,
        mapping: Mapping,
        prefix: str,
        mm_attention_backend: str | None,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attn = VisionAttention(
            embed_dim=hidden_size,
            num_heads=config.num_attention_heads,
            mapping=mapping,
            qkv_bias=True,
            proj_bias=True,
            prefix=f"{prefix}.attn",
            mm_attention_backend=mm_attention_backend,
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.mlp = RadioMLP(hidden_size, config.intermediate_size)

    def forward(
        self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        ).squeeze(0)
        return x + self.mlp(self.norm2(x))


class RadioVisionModel(nn.Module):
    """The C-RADIO ViT; ``config.final_norm`` adds a LayerNorm after the last block."""

    def __init__(
        self,
        config: RadioVisionConfig,
        mapping: Mapping,
        *,
        mm_attention_backend: str | None,
        prefix: str,
    ) -> None:
        super().__init__()
        if mm_attention_backend == "flashinfer_cudnn":
            raise NotImplementedError(
                "C-RADIO with the flashinfer_cudnn encoder attention backend"
            )
        if config.video_temporal_patch_size < 2 or not config.separate_video_embedder:
            raise NotImplementedError(
                "C-RADIO video without a separate tubelet embedder"
            )
        patch_dim = 3 * config.patch_size**2
        hidden_size = config.hidden_size
        self.temporal_patch_size = config.video_temporal_patch_size
        # The planner's largest image grid, two rows of the patch budget, bounds a video pass.
        self.video_patches_per_pass = 2 * config.image_patch_budget
        self.grid_size = config.position_grid_size
        self.num_prefix_tokens = config.num_prefix_tokens
        self.embedder = nn.Linear(patch_dim, hidden_size, bias=False)
        self.video_embedder = nn.Linear(
            self.temporal_patch_size * patch_dim, hidden_size, bias=False
        )
        self.pos_embed = nn.Parameter(
            torch.empty(1, self.grid_size * self.grid_size, hidden_size)
        )
        self.cls_token = nn.Parameter(torch.empty(self.num_prefix_tokens, hidden_size))
        self.blocks = nn.ModuleList(
            RadioBlock(config, mapping, f"{prefix}.blocks.{i}", mm_attention_backend)
            for i in range(config.num_hidden_layers)
        )
        self.norm = (
            nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
            if config.final_norm
            else None
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.embedder.weight.dtype

    def position_embedding(self, rows: int, cols: int) -> torch.Tensor:
        """The ``[rows * cols, hidden]`` embedding of a ``rows x cols`` patch grid.

        Equals resizing the grid to ``max(rows, cols)`` square and cropping, but
        an upscale computes only the cropped outputs, so a thin grid does not
        allocate the full square.
        """
        side = max(rows, cols)
        grid = self.pos_embed.view(1, self.grid_size, self.grid_size, -1)
        # The ref engine resizes this channels-last view; the NCHW kernel rounds differently.
        grid = grid.permute(0, 3, 1, 2).float()
        if side < self.grid_size:
            grid = F.interpolate(
                grid, size=(side, side), mode="bilinear", align_corners=False
            )
        elif side > self.grid_size:
            # The square resize's scale with only its top-left rows x cols outputs.
            scale = side / self.grid_size
            grid = torch.ops.aten.upsample_bilinear2d(
                grid, [rows, cols], False, scale, scale
            )
        crop = grid[0, :, :rows, :cols].to(self.pos_embed.dtype)
        return crop.flatten(1).transpose(0, 1)

    def embed_images(
        self, patches: torch.Tensor, grids: list[tuple[int, int]]
    ) -> torch.Tensor:
        """Embed images given as concatenated ``[sum h*w, 3*p*p]`` patches."""
        return self.encode(self.image_sequences(patches, grids))

    def image_sequences(
        self, patches: torch.Tensor, grids: list[tuple[int, int]]
    ) -> list[torch.Tensor]:
        """Each image's patch embeddings plus its position embedding."""
        embedded = self.embedder(patches.to(self.dtype))
        sequences = []
        start = 0
        for rows, cols in grids:
            image = embedded[start : start + rows * cols]
            sequences.append(image + self.position_embedding(rows, cols))
            start += rows * cols
        return sequences

    def embed_video(
        self, patches: torch.Tensor, frames: int, rows: int, cols: int
    ) -> torch.Tensor:
        """Embed ``frames`` frames as tubelets; returns ``[tubelets * rows * cols, hidden]``.

        A short last tubelet repeats the final frame. Tubelets attend only within
        themselves, so they run in passes no larger than the largest image.
        """
        frame_patches = patches.to(self.dtype).view(frames, rows * cols, -1)
        pad = -frames % self.temporal_patch_size
        if pad:
            frame_patches = torch.cat(
                [frame_patches, frame_patches[-1:].expand(pad, -1, -1)]
            )
        tubelets = frame_patches.shape[0] // self.temporal_patch_size
        # Each tubelet patch concatenates its frames' patches in frame order.
        tubelet_patches = (
            frame_patches.view(tubelets, self.temporal_patch_size, rows * cols, -1)
            .transpose(1, 2)
            .reshape(tubelets, rows * cols, -1)
        )
        embedded = self.video_embedder(tubelet_patches) + self.position_embedding(
            rows, cols
        )
        per_pass = max(1, self.video_patches_per_pass // (rows * cols))
        return torch.cat(
            [self.encode(list(chunk.unbind(0))) for chunk in embedded.split(per_pass)]
        )

    def encode(self, sequences: list[torch.Tensor]) -> torch.Tensor:
        """Run the blocks over embedded sequences; drop each one's prefix tokens."""
        x, lengths = self.pack(sequences)
        cu_seqlens = torch.tensor(
            [0, *accumulate(lengths)], dtype=torch.int32, device=x.device
        )
        return self.unpack(self.forward_blocks(x, cu_seqlens, max(lengths)), lengths)

    def pack(self, sequences: list[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
        """Prepend the prefix tokens to each sequence and concatenate them."""
        lengths = [self.num_prefix_tokens + len(seq) for seq in sequences]
        x = torch.cat([part for seq in sequences for part in (self.cls_token, seq)])
        return x, lengths

    def forward_blocks(
        self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
    ) -> torch.Tensor:
        """The blocks and final norm over packed sequences."""
        for block in self.blocks:
            x = block(x, cu_seqlens, max_seqlen)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def unpack(self, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        """Drop each packed sequence's prefix tokens."""
        return torch.cat([seq[self.num_prefix_tokens :] for seq in x.split(lengths)])

    def load_weight(self, name: str, loaded: torch.Tensor) -> None:
        """Load one checkpoint tensor named relative to the RADIO ViT."""
        name = name.removeprefix("patch_generator.")
        name = name.replace("cls_token.token", "cls_token")
        name = name.replace(".attn.qkv.", ".attn.qkv_proj.")
        param = self.get_parameter(name)
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded)


@dataclass
class RadioEncoderBatch:
    """Images packed for the blocks: their prefixed sequences and patch grids."""

    tokens: torch.Tensor
    lengths: list[int]
    grids: list[tuple[int, int]]

    @property
    def input_tensors(self) -> dict[str, torch.Tensor]:
        return {"tokens": self.tokens}

    @property
    def encoder_output_tokens(self) -> list[int]:
        return self.lengths

    @property
    def metadata_sequences(self) -> list[int]:
        return [1] * len(self.lengths)

    def num_items(self) -> int:
        return len(self.lengths)

    def select(self, indices: list[int]) -> "RadioEncoderBatch":
        """Sub-batch at ``indices``, preserving order."""
        sequences = self.tokens.split(self.lengths)
        return RadioEncoderBatch(
            torch.cat([sequences[i] for i in indices]),
            [self.lengths[i] for i in indices],
            [self.grids[i] for i in indices],
        )


@dataclass
class RadioEncoderCudaGraphAdapter:
    """Encoder CUDA graph adapter capturing the RADIO blocks over packed images.

    Embedding runs eagerly before the graph, since the position embedding is
    resized per image, and ``post_encode`` maps the unpacked patch features
    to the model's output after it. Budgets count packed rows, prefix tokens
    included; each image is one attention sequence.
    """

    tower: RadioVisionModel
    post_encode: Callable[[torch.Tensor, list[tuple[int, int]]], torch.Tensor]
    max_image_tokens: int
    capture_tp_size: int
    capture_tp_group: Any | None
    modality_name: str = "vision"

    @property
    def device(self) -> torch.device:
        return self.tower.embedder.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.tower.dtype

    def batch_from_items(self, items: list[Any]) -> RadioEncoderBatch:
        grids = [tuple(item.image_grid_hw.reshape(-1).tolist()) for item in items]
        patches = torch.cat([item.feature.to(self.device) for item in items])
        tokens, lengths = self.tower.pack(self.tower.image_sequences(patches, grids))
        if max(lengths) > self.max_image_tokens:
            raise ValueError(
                f"RADIO image of {max(lengths)} tokens exceeds the "
                f"{self.max_image_tokens} the encoder cudagraph bounds attention by"
            )
        return RadioEncoderBatch(tokens, lengths, grids)

    def capture_batch_for_budget(
        self,
        encoder_output_token_budget: int,
        _max_batch_size: int,
        _metadata_sequence_budget: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RadioEncoderBatch:
        hidden_size = self.tower.embedder.out_features
        tokens = torch.zeros(
            encoder_output_token_budget, hidden_size, device=device, dtype=dtype
        )
        full, rest = divmod(encoder_output_token_budget, self.max_image_tokens)
        lengths = [self.max_image_tokens] * full + ([rest] if rest else [])
        return RadioEncoderBatch(tokens, lengths, [])

    def prepare_metadata(
        self,
        batch: RadioEncoderBatch,
        encoder_output_token_budget: int | None,
        metadata_sequence_budget: int,
    ) -> dict[str, Any]:
        cu_seqlens = torch.tensor(
            [0, *accumulate(batch.lengths)], dtype=torch.int32, device=self.device
        )
        if encoder_output_token_budget is None:
            return {"cu_seqlens": cu_seqlens, "max_seqlen": max(batch.lengths)}
        return {
            "cu_seqlens": pad_cu_seqlens(
                cu_seqlens,
                sum(batch.lengths),
                encoder_output_token_budget,
                metadata_sequence_budget,
                self.modality_name,
            ),
            "max_seqlen": min(
                encoder_output_token_budget,
                max(self.max_image_tokens, PADDING_SEQUENCE_ROWS),
            ),
        }

    def forward(
        self, input_tensors: dict[str, torch.Tensor], metadata: dict[str, Any]
    ) -> torch.Tensor:
        return self.tower.forward_blocks(
            input_tensors["tokens"], metadata["cu_seqlens"], metadata["max_seqlen"]
        )

    def postprocess(
        self, encoder_outs: list[torch.Tensor], batch: RadioEncoderBatch
    ) -> torch.Tensor:
        features = torch.cat(
            [out[self.tower.num_prefix_tokens :] for out in encoder_outs]
        )
        return self.post_encode(features, batch.grids)
