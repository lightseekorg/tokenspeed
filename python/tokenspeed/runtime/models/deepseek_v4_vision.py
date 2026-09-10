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

"""DeepSeek V4 vision encoder, aligner, and image embeddings."""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from tokenspeed.runtime.configs.deepseek_v4_config import (
    DeepseekV4Config,
)
from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.layers.attention.mm_encoder_attention import (
    VIT_CUDNN_WORKSPACE_BYTES,
    VisionAttention,
)
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem


@lru_cache(maxsize=16)
def get_vision_cos_sin(
    n_h: int,
    n_w: int,
    dim: int,
    theta: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1).to(device), freqs.sin().unsqueeze(1).to(device)


def apply_vision_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    _x_shape: torch.Size,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = position_embeddings
    outputs = []
    for x in (q, k):
        x1, x2 = x.float().chunk(2, dim=-1)
        outputs.append(
            torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)
        )
    return outputs[0], outputs[1]


class DeepseekV4VisionRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class DeepseekV4VisionPatchEmbed(nn.Module):
    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        patch_size = config.vision_patch_size
        self.proj = nn.Linear(3 * patch_size**2, config.vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class DeepseekV4VisionMLP(nn.Module):
    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.w1 = nn.Linear(
            config.vision_dim,
            2 * config.vision_inter_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            config.vision_inter_dim,
            config.vision_dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class DeepseekV4VisionBlock(nn.Module):
    def __init__(
        self,
        config: DeepseekV4Config,
        mapping: Mapping,
        mm_attention_backend: str | None,
        workspace_buffer: torch.Tensor | None,
    ) -> None:
        super().__init__()
        dim = config.vision_dim
        self.norm1 = DeepseekV4VisionRMSNorm(dim, eps=1e-6)
        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=config.vision_n_heads,
            head_size=dim // config.vision_n_heads,
            mapping=mapping,
            quant_config=None,
            prefix="",
            proj_bias=True,
            qkv_bias=True,
            customized_position_embedding_applier=apply_vision_rotary,
            position_embedding_mode=None,
            workspace_buffer=workspace_buffer,
            mm_attention_backend=mm_attention_backend,
        )
        self.norm2 = DeepseekV4VisionRMSNorm(dim, eps=1e-6)
        self.mlp = DeepseekV4VisionMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        attn_output = self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            position_embeddings=(cos, sin),
            rotary_pos_emb_cos=None,
            rotary_pos_emb_sin=None,
            max_seqlen=x.shape[0],
            sequence_lengths=sequence_lengths,
        )
        x = x + attn_output.squeeze(0)
        return x + self.mlp(self.norm2(x))


class DeepseekV4VisionTower(nn.Module):
    def __init__(
        self,
        config: DeepseekV4Config,
        mapping: Mapping,
        mm_attention_backend: str | None,
    ) -> None:
        super().__init__()
        self.rope_dim = config.vision_dim // config.vision_n_heads // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = DeepseekV4VisionPatchEmbed(config)
        self.mm_attention_backend = mm_attention_backend
        self.local_dim = config.vision_dim // mapping.vision.tp_size
        workspace_buffer = None
        if mm_attention_backend == "flashinfer_cudnn":
            workspace_buffer = torch.empty(
                VIT_CUDNN_WORKSPACE_BYTES,
                dtype=torch.uint8,
                device=torch.device("cuda", torch.cuda.current_device()),
            )
        self.blocks = nn.ModuleList(
            [
                DeepseekV4VisionBlock(
                    config, mapping, mm_attention_backend, workspace_buffer
                )
                for _ in range(config.vision_n_layers)
            ]
        )
        self.norm = DeepseekV4VisionRMSNorm(config.vision_dim, eps=1e-6)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(
            n_h, n_w, self.rope_dim, self.rope_theta, x.device
        )
        cu_seqlens = torch.tensor([0, x.shape[0]], dtype=torch.int32, device=x.device)
        sequence_lengths = None
        if self.mm_attention_backend == "flashinfer_cudnn":
            sequence_lengths = cu_seqlens[1:]
            cu_seqlens = (cu_seqlens * self.local_dim).repeat(3)
        for block in self.blocks:
            x = block(x, cos, sin, cu_seqlens, sequence_lengths)
        return self.norm(x)


class DeepseekV4VisionAligner(nn.Module):
    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.downsample_ratio = config.vision_downsample_ratio
        in_dim = config.vision_dim * self.downsample_ratio**2
        hidden_size = config.hidden_size
        self.w1 = nn.Linear(in_dim, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        ratio = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % ratio, 0, -n_h % ratio))
        x = F.unfold(x.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))


class DeepseekV4Vision(nn.Module):
    """ViT + aligner + learned image sentinel vectors."""

    def __init__(
        self,
        config: DeepseekV4Config,
        mapping: Mapping,
        mm_attention_backend: str | None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vision = DeepseekV4VisionTower(config, mapping, mm_attention_backend)
        self.aligner = DeepseekV4VisionAligner(config)
        hidden_size = config.hidden_size
        self.image_start = nn.Parameter(torch.empty(hidden_size))
        self.image_end = nn.Parameter(torch.empty(hidden_size))
        self.image_newline = nn.Parameter(torch.empty(hidden_size))
        self.image_pad = nn.Parameter(torch.empty(hidden_size))

    def encode_image(
        self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int
    ) -> torch.Tensor:
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def embed_one(self, item: MultimodalDataItem) -> torch.Tensor:
        data = item.model_specific_data
        n_vit_h = int(data["n_vit_h"])
        n_vit_w = int(data["n_vit_w"])
        block_length = sum(end - start + 1 for start, end in item.offsets)
        types = data["types"][-block_length:].to(
            device=self.image_start.device, dtype=torch.int64
        )
        perm = data["perm"].to(device=self.image_start.device, dtype=torch.int64)
        patches = item.feature.to(
            device=self.image_start.device, dtype=self.image_start.dtype
        )
        embeds = self.encode_image(patches, n_vit_h, n_vit_w)[perm]
        block = torch.stack(
            [
                self.image_start,
                self.image_pad,
                self.image_pad,
                self.image_newline,
                self.image_end,
            ]
        )[types]
        image_mask = types == 2
        num_image_tokens = int(image_mask.sum())
        if embeds.size(0) != num_image_tokens:
            raise ValueError(
                f"aligner produced {embeds.size(0)} tokens but the block has "
                f"{num_image_tokens} IMAGE slots"
            )
        block[image_mask] = embeds
        return block

    def embed_media(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        if not items:
            return self.image_start.new_empty((0, self.image_start.numel()))
        return torch.cat([self.embed_one(item) for item in items], dim=0)

    def make_image_warmup_items(self) -> list[MultimodalDataItem]:
        patch_size = self.config.vision_patch_size
        n_vit_h = self.config.vision_downsample_ratio
        n_vit_w = n_vit_h
        types = torch.tensor(
            [1, 1, 1, 0, 2, 1, 3, 1, 4],
            dtype=torch.int64,
        )
        patches = torch.zeros(
            (n_vit_h * n_vit_w, 3, patch_size, patch_size),
            dtype=self.vision.patch_embed.proj.weight.dtype,
        )
        return [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                offsets=[(0, int(types.numel()) - 1)],
                feature=patches,
                model_specific_data={
                    "n_vit_h": torch.tensor(n_vit_h, dtype=torch.int64),
                    "n_vit_w": torch.tensor(n_vit_w, dtype=torch.int64),
                    "types": types,
                    "perm": torch.tensor([0], dtype=torch.int64),
                },
            )
        ]
