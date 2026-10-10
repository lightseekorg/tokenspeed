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

from itertools import accumulate

import torch
import torch.nn.functional as F
from torch import nn

from tokenspeed.runtime.configs.nemotron_omni_config import RadioVisionConfig
from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.layers.attention.mm_encoder_attention import VisionAttention
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader


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
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = VisionAttention(
            embed_dim=hidden_size,
            num_heads=config.num_attention_heads,
            mapping=mapping,
            qkv_bias=True,
            proj_bias=True,
            prefix=f"{prefix}.attn",
            mm_attention_backend=mm_attention_backend,
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = RadioMLP(hidden_size, config.intermediate_size)

    def forward(
        self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        ).squeeze(0)
        return x + self.mlp(self.norm2(x))


class RadioVisionModel(nn.Module):
    """The C-RADIO ViT; ``final_norm`` adds a LayerNorm after the last block."""

    def __init__(
        self,
        config: RadioVisionConfig,
        mapping: Mapping,
        *,
        final_norm: bool,
        mm_attention_backend: str | None,
        prefix: str,
    ) -> None:
        super().__init__()
        patch_dim = 3 * config.patch_size**2
        hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.video_temporal_patch_size
        self.grid_size = config.position_grid_size
        self.num_prefix_tokens = config.num_prefix_tokens
        self.embedder = nn.Linear(patch_dim, hidden_size, bias=False)
        self.video_embedder = (
            nn.Linear(self.temporal_patch_size * patch_dim, hidden_size, bias=False)
            if self.temporal_patch_size > 1
            else None
        )
        self.pos_embed = nn.Parameter(
            torch.empty(1, self.grid_size * self.grid_size, hidden_size)
        )
        self.cls_token = nn.Parameter(torch.empty(self.num_prefix_tokens, hidden_size))
        self.blocks = nn.ModuleList(
            RadioBlock(config, mapping, f"{prefix}.blocks.{i}", mm_attention_backend)
            for i in range(config.num_hidden_layers)
        )
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6) if final_norm else None

    @property
    def dtype(self) -> torch.dtype:
        return self.embedder.weight.dtype

    def position_embedding(self, rows: int, cols: int) -> torch.Tensor:
        """The ``[rows * cols, hidden]`` embedding of a ``rows x cols`` patch grid."""
        side = max(rows, cols)
        grid = self.pos_embed.view(1, self.grid_size, self.grid_size, -1)
        grid = grid.permute(0, 3, 1, 2)
        if side != self.grid_size:
            grid = F.interpolate(
                grid.float(), size=(side, side), mode="bilinear", align_corners=False
            ).to(grid.dtype)
        return grid[0, :, :rows, :cols].flatten(1).transpose(0, 1)

    def embed_images(
        self, patches: torch.Tensor, grids: list[tuple[int, int]]
    ) -> torch.Tensor:
        """Embed images given as concatenated ``[sum h*w, 3*p*p]`` patches."""
        embedded = self.embedder(patches.to(self.dtype))
        sequences = []
        start = 0
        for rows, cols in grids:
            image = embedded[start : start + rows * cols]
            sequences.append(image + self.position_embedding(rows, cols))
            start += rows * cols
        return self.encode(sequences)

    def embed_video(
        self, patches: torch.Tensor, frames: int, rows: int, cols: int
    ) -> torch.Tensor:
        """Embed ``frames`` frames as tubelets; returns ``[tubelets * rows * cols, hidden]``.

        A short last tubelet repeats the final frame.
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
        return self.encode(list(embedded.unbind(0)))

    def encode(self, sequences: list[torch.Tensor]) -> torch.Tensor:
        """Run the blocks over embedded sequences; drop each one's prefix tokens."""
        prefix = self.cls_token.to(sequences[0].dtype)
        lengths = [self.num_prefix_tokens + len(seq) for seq in sequences]
        x = torch.cat([part for seq in sequences for part in (prefix, seq)])
        cu_seqlens = torch.tensor(
            [0, *accumulate(lengths)], dtype=torch.int32, device=x.device
        )
        max_seqlen = max(lengths)
        for block in self.blocks:
            x = block(x, cu_seqlens, max_seqlen)
        if self.norm is not None:
            x = self.norm(x)
        keep = torch.cat(
            [
                torch.arange(start + self.num_prefix_tokens, end, device=x.device)
                for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())
            ]
        )
        return x[keep]

    def load_weight(self, name: str, loaded: torch.Tensor) -> None:
        """Load one checkpoint tensor named relative to the RADIO ViT."""
        name = name.removeprefix("patch_generator.")
        name = name.replace("cls_token.token", "cls_token")
        name = name.replace(".attn.qkv.", ".attn.qkv_proj.")
        param = dict(self.named_parameters())[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded)
