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

"""DeepSeek V4.1 vision encoder, aligner, and image embeddings."""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from tokenspeed.runtime.configs.deepseek_v41_config import DeepseekV41Config
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


class DeepseekV41VisionPatchEmbed(nn.Module):
    def __init__(self, config: DeepseekV41Config) -> None:
        super().__init__()
        vision = config.vision_config
        patch_size = vision.patch_size
        self.proj = nn.Linear(3 * patch_size**2, vision.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class DeepseekV41VisionMLP(nn.Module):
    def __init__(self, config: DeepseekV41Config) -> None:
        super().__init__()
        vision = config.vision_config
        self.w1 = nn.Linear(
            vision.hidden_size,
            2 * vision.intermediate_size,
            bias=False,
        )
        self.w2 = nn.Linear(
            vision.intermediate_size,
            vision.hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class DeepseekV41VisionBlock(nn.Module):
    def __init__(
        self,
        config: DeepseekV41Config,
        mapping: Mapping,
        mm_attention_backend: str | None,
        workspace_buffer: torch.Tensor | None,
    ) -> None:
        super().__init__()
        vision = config.vision_config
        dim = vision.hidden_size
        self.norm1 = nn.RMSNorm(
            dim, eps=1e-6, elementwise_affine=True, device=None, dtype=torch.float32
        )
        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=vision.num_attention_heads,
            head_size=dim // vision.num_attention_heads,
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
        self.norm2 = nn.RMSNorm(
            dim, eps=1e-6, elementwise_affine=True, device=None, dtype=torch.float32
        )
        self.mlp = DeepseekV41VisionMLP(config)

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


class DeepseekV41VisionTower(nn.Module):
    def __init__(
        self,
        config: DeepseekV41Config,
        mapping: Mapping,
        mm_attention_backend: str | None,
    ) -> None:
        super().__init__()
        vision = config.vision_config
        self.rope_dim = vision.hidden_size // vision.num_attention_heads // 2
        self.rope_theta = vision.rope_theta
        self.patch_embed = DeepseekV41VisionPatchEmbed(config)
        self.mm_attention_backend = mm_attention_backend
        self.local_dim = vision.hidden_size // mapping.vision.tp_size
        workspace_buffer = None
        if mm_attention_backend == "flashinfer_cudnn":
            workspace_buffer = torch.empty(
                VIT_CUDNN_WORKSPACE_BYTES,
                dtype=torch.uint8,
                device=torch.device("cuda", torch.cuda.current_device()),
            )
        self.blocks = nn.ModuleList(
            [
                DeepseekV41VisionBlock(
                    config, mapping, mm_attention_backend, workspace_buffer
                )
                for _ in range(vision.num_hidden_layers)
            ]
        )
        self.norm = nn.RMSNorm(
            vision.hidden_size,
            eps=1e-6,
            elementwise_affine=True,
            device=None,
            dtype=torch.float32,
        )

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


class DeepseekV41VisionAligner(nn.Module):
    def __init__(self, config: DeepseekV41Config) -> None:
        super().__init__()
        vision = config.vision_config
        self.downsample_ratio = vision.downsample_ratio
        in_dim = vision.hidden_size * self.downsample_ratio**2
        hidden_size = config.text_config.hidden_size
        self.w1 = nn.Linear(in_dim, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        ratio = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % ratio, 0, -n_h % ratio))
        x = F.unfold(x.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))


class DeepseekV41Vision(nn.Module):
    """ViT + aligner + learned image sentinel vectors."""

    def __init__(
        self,
        config: DeepseekV41Config,
        mapping: Mapping,
        mm_attention_backend: str | None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vision = DeepseekV41VisionTower(config, mapping, mm_attention_backend)
        self.aligner = DeepseekV41VisionAligner(config)
        hidden_size = config.text_config.hidden_size
        self.image_start = nn.Parameter(torch.empty(hidden_size))
        self.image_end = nn.Parameter(torch.empty(hidden_size))
        self.image_newline = nn.Parameter(torch.empty(hidden_size))

    def embed_one(self, item: MultimodalDataItem) -> torch.Tensor:
        data = item.model_specific_data
        n_vit_h, n_vit_w = data["vit_grid"].reshape(-1).tolist()
        n_llm_h, n_llm_w = data["llm_grid"].reshape(-1).tolist()
        ratio = self.aligner.downsample_ratio
        if (n_llm_h, n_llm_w) != (
            (n_vit_h + ratio - 1) // ratio,
            (n_vit_w + ratio - 1) // ratio,
        ):
            raise ValueError("V4.1 vit_grid and llm_grid disagree")
        types = data["types"].reshape(-1)
        expected = torch.tensor([0] + ([1] * n_llm_w + [2]) * n_llm_h + [3])
        if not torch.equal(types.cpu(), expected):
            raise ValueError("V4.1 image types do not match llm_grid")
        if item.feature.shape[0] != n_vit_h * n_vit_w:
            raise ValueError("V4.1 patch count does not match vit_grid")
        patches = item.feature.to(
            device=self.image_start.device, dtype=self.image_start.dtype
        )
        embeds = self.vision(patches, n_vit_h, n_vit_w)
        embeds = self.aligner(embeds, n_vit_h, n_vit_w)
        types = types.to(device=embeds.device, dtype=torch.int64)
        block = torch.stack(
            [self.image_start, self.image_start, self.image_newline, self.image_end]
        )[types]
        block[types == 1] = embeds
        return block

    def embed_media(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return torch.cat([self.embed_one(item) for item in items], dim=0)

    def make_image_warmup_items(self) -> list[MultimodalDataItem]:
        vision = self.config.vision_config
        patch_size = vision.patch_size
        n_vit_h = vision.downsample_ratio
        n_vit_w = n_vit_h
        types = torch.tensor(
            [0, 1, 2, 3],
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
                    "vit_grid": torch.tensor([n_vit_h, n_vit_w], dtype=torch.int64),
                    "llm_grid": torch.tensor([1, 1], dtype=torch.int64),
                    "types": types,
                },
            )
        ]
