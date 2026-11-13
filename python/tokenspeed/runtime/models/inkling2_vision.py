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

"""BF16 Perception Encoder with native checkpoint parameter names.

Inputs are normalized image patches, not decoded RGB images. Each image has a
separate class token and bidirectional attention segment. Native patches use
pixel-major RGB order; the projector joins adjacent spatial patches without
averaging. Weight sharding, video, and image preprocessing are not provided.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from itertools import accumulate

import torch
import torch.nn.functional as F
from torch import nn

from tokenspeed.runtime.configs.inkling2 import Inkling2VisionConfig
from tokenspeed.runtime.layers.attention.mm_encoder_attention import _resolve_backend
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem
from tokenspeed.runtime.utils.host_sync import host_to_device


def resolve_varlen_encoder_attention_backend(
    name: str | None,
) -> Callable[..., torch.Tensor]:
    """Resolve attention over ordinary token-offset cumulative lengths.

    Unlike the cuDNN path, this contract needs no packed element offsets or
    caller-owned workspace. Unknown or platform-incompatible names raise.
    """
    if name == "flashinfer_cudnn":
        raise NotImplementedError(
            "Varlen encoder attention does not prepare cuDNN packed offsets "
            "and workspace"
        )
    return _resolve_backend(name)


def varlen_encoder_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    lengths: list[int],
    softmax_scale: float,
    backend: str | None,
) -> torch.Tensor:
    """Attend independent bidirectional segments with ordinary token offsets.

    Args:
        q, k, v: Packed token/head/channel tensors of one dtype and device.
        cu_seqlens: Cumulative token offsets, including zero and the total.
        lengths: CPU segment lengths, including any per-image class token.
        softmax_scale: Explicit multiplier of the QK scores.
        backend: Existing multimodal backend name, or the platform default.

    Returns:
        Packed attention output with q's shape and dtype. CPU diagnostics use
        the Torch reference for automatic or portable Triton selection.
    """
    if not q.is_cuda and backend in (None, "triton_attn"):
        return torch.cat(
            [
                torch.nn.functional.scaled_dot_product_attention(
                    qi.transpose(0, 1).unsqueeze(0),
                    ki.transpose(0, 1).unsqueeze(0),
                    vi.transpose(0, 1).unsqueeze(0),
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=softmax_scale,
                )
                .squeeze(0)
                .transpose(0, 1)
                for qi, ki, vi in zip(
                    q.split(lengths), k.split(lengths), v.split(lengths)
                )
            ]
        )
    fn = resolve_varlen_encoder_attention_backend(backend)
    return fn(
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        bsz=len(lengths),
        seq_len=max(lengths),
        softmax_scale=softmax_scale,
        max_seqlen=max(lengths),
    )


def _position_embeddings(
    grid: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    heights: torch.Tensor,
    widths: torch.Tensor,
) -> torch.Tensor:
    """Interpolate learned grid centers in FP32, then materialize BF16 rows."""

    def weights(
        indices: torch.Tensor, targets: torch.Tensor, size: int
    ) -> torch.Tensor:
        center = (size / targets.float()) * (indices.float() + 0.5)
        distance = torch.arange(size, device=grid.device)[None, :] + 0.5
        values = (1 - (distance - center[:, None]).abs()).clamp(min=0)
        return values / values.sum(dim=1, keepdim=True)

    return torch.einsum(
        "ni,nj,ijd->nd",
        weights(y, heights, grid.shape[0]),
        weights(x, widths, grid.shape[1]),
        grid.float(),
    ).to(grid.dtype)


def _rotary(
    q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate adjacent channel pairs: X in the first half, Y in the second."""
    dim = q.shape[-1]
    frequencies = 1.0 / (
        10000 ** (torch.arange(0, dim, 4, device=q.device, dtype=torch.float32) / dim)
    )
    angles = torch.cat(
        (positions[:, 1, None] * frequencies, positions[:, 0, None] * frequencies),
        dim=-1,
    )
    phase = torch.polar(torch.ones_like(angles), angles)
    cos, sin = phase.real[:, None, :], phase.imag[:, None, :]

    def rotate(value: torch.Tensor) -> torch.Tensor:
        pairs = value.float().reshape(*value.shape[:-1], -1, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        return (
            torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
            .flatten(-2)
            .to(value.dtype)
        )

    return rotate(q), rotate(k)


class Inkling2VisionAttention(nn.Module):
    def __init__(
        self, width: int, heads: int, mm_attention_backend: str | None
    ) -> None:
        super().__init__()
        self.mm_attention_backend = mm_attention_backend
        self.num_heads = heads
        self.head_dim = width // heads
        self.q_proj = nn.Linear(width, width, bias=True, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(width, width, bias=True, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(width, width, bias=True, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(width, width, bias=True, dtype=torch.bfloat16)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        lengths: list[int],
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        shape = (-1, self.num_heads, self.head_dim)
        q, k, v = (
            projection(hidden).reshape(shape)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        )
        q, k = _rotary(q, k, positions)
        output = varlen_encoder_attention(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            lengths=lengths,
            softmax_scale=self.head_dim**-0.5,
            backend=self.mm_attention_backend,
        )
        return self.out_proj(output.reshape(hidden.shape))


class Inkling2VisionMLP(nn.Module):
    def __init__(self, width: int, intermediate_size: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(width, intermediate_size, bias=True, dtype=torch.bfloat16)
        self.c_proj = nn.Linear(
            intermediate_size, width, bias=True, dtype=torch.bfloat16
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(hidden), approximate="none"))


class Inkling2VisionLayerScale(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(width, dtype=torch.bfloat16))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden * self.gamma


class Inkling2VisionLayer(nn.Module):
    def __init__(
        self, config: Inkling2VisionConfig, mm_attention_backend: str | None
    ) -> None:
        super().__init__()
        width, eps = config.hidden_size, config.layer_norm_eps
        self.ln_1 = nn.LayerNorm(
            width, eps=eps, elementwise_affine=True, bias=True, dtype=torch.bfloat16
        )
        self.self_attn = Inkling2VisionAttention(
            width, config.num_attention_heads, mm_attention_backend
        )
        self.ln_2 = nn.LayerNorm(
            width, eps=eps, elementwise_affine=True, bias=True, dtype=torch.bfloat16
        )
        self.mlp = Inkling2VisionMLP(width, int(width * config.mlp_ratio))
        self.ls_1 = Inkling2VisionLayerScale(width)
        self.ls_2 = Inkling2VisionLayerScale(width)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        lengths: list[int],
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        hidden = hidden + self.ls_1(
            self.self_attn(self.ln_1(hidden), positions, lengths, cu_seqlens)
        )
        return hidden + self.ls_2(self.mlp(self.ln_2(hidden)))


class Inkling2VisionEmbeddings(nn.Module):
    def __init__(self, config: Inkling2VisionConfig) -> None:
        super().__init__()
        width = config.hidden_size
        self.patch_embedding = nn.Linear(
            3 * config.native_patch_size**2, width, bias=False, dtype=torch.bfloat16
        )
        self.position_embedding = nn.Parameter(
            torch.empty(config.posemb_grid_size**2, width, dtype=torch.bfloat16)
        )
        self.class_embedding = nn.Parameter(torch.empty(width, dtype=torch.bfloat16))
        self.cls_position_embedding = nn.Parameter(
            torch.empty(width, dtype=torch.bfloat16)
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.patch_embedding(pixels)


class Inkling2VisionBackbone(nn.Module):
    def __init__(
        self, config: Inkling2VisionConfig, mm_attention_backend: str | None
    ) -> None:
        super().__init__()
        self.embeddings = Inkling2VisionEmbeddings(config)
        self.ln_pre = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            elementwise_affine=True,
            bias=True,
            dtype=torch.bfloat16,
        )
        self.layers = nn.ModuleList(
            [
                Inkling2VisionLayer(config, mm_attention_backend)
                for _ in range(config.num_hidden_layers)
            ]
        )


class Inkling2VisionEncoder(nn.Module):
    """Encode preprocessed images without mixing their attention segments.

    Args:
        vision_config: Explicit PE geometry from the checkpoint.
        text_hidden_size: Output width of the multimodal projector.
        mm_attention_backend: Shared multimodal backend, or automatic selection.

    Forward takes BF16 patches [N, temporal, side, side, 3] and a CPU integer
    grid [images, 3]. The grid is [1, rows, columns], and side is native patch
    size times merge size. Returns BF16 [N, text_hidden_size] in image order.
    """

    def __init__(
        self,
        vision_config: Inkling2VisionConfig,
        text_hidden_size: int,
        mm_attention_backend: str | None,
    ) -> None:
        super().__init__()
        if mm_attention_backend is not None:
            resolve_varlen_encoder_attention_backend(mm_attention_backend)
        for name in (
            "temporal_patch_size",
            "native_patch_size",
            "adapter_merge_size",
            "projection_hidden_dim",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "posemb_grid_size",
        ):
            value = getattr(vision_config, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"vision_config.{name} must be a positive integer")
        if type(text_hidden_size) is not int or text_hidden_size <= 0:
            raise ValueError("text_hidden_size must be a positive integer")
        for name in ("mlp_ratio", "layer_norm_eps"):
            value = getattr(vision_config, name)
            if (
                type(value) not in (float, int)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"vision_config.{name} must be finite and positive")
        width, heads = vision_config.hidden_size, vision_config.num_attention_heads
        if width % heads or width // heads not in (4, 8, 16, 32, 64, 128):
            raise ValueError(
                "Vision head dimension must be one of 4, 8, 16, 32, 64, 128"
            )
        intermediate_size = width * vision_config.mlp_ratio
        if not math.isfinite(intermediate_size) or intermediate_size != int(
            intermediate_size
        ):
            raise ValueError("Vision MLP intermediate size must be an integer")
        if type(vision_config.projector_bias) is not bool:
            raise ValueError("vision_config.projector_bias must be a boolean")
        self.config = vision_config
        self.patch_size = vision_config.native_patch_size
        self.merge_size = vision_config.adapter_merge_size
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.backbone = Inkling2VisionBackbone(vision_config, mm_attention_backend)
        self.proj_in = nn.Linear(
            self.merge_size**2 * width,
            vision_config.projection_hidden_dim,
            bias=vision_config.projector_bias,
            dtype=torch.bfloat16,
        )
        self.proj_out = nn.Linear(
            vision_config.projection_hidden_dim,
            text_hidden_size,
            bias=vision_config.projector_bias,
            dtype=torch.bfloat16,
        )
        for name, values in (
            ("_clip_mean", (0.48145466, 0.4578275, 0.40821073)),
            ("_clip_std", (0.26862954, 0.26130258, 0.27577711)),
        ):
            self.register_buffer(
                name,
                torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, 1, 3),
                persistent=False,
            )

    def make_warmup_items(self) -> list[MultimodalDataItem]:
        """Return one independent image token using the configured patch geometry."""
        side = self.patch_size * self.merge_size
        return [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                offsets=[(0, 0)],
                feature=torch.zeros(
                    (1, self.temporal_patch_size, side, side, 3),
                    dtype=torch.bfloat16,
                    device="cpu",
                ),
                model_specific_data={
                    "vision_grid_thw": torch.tensor(
                        [[1, 1, 1]], dtype=torch.int64, device="cpu"
                    )
                },
            )
        ]

    def _preprocess_patches(self, patches: torch.Tensor) -> torch.Tensor:
        rgb = patches * self._clip_std.to(patches.dtype) + self._clip_mean.to(
            patches.dtype
        )
        pixels = (2.0 * rgb - 1.0).mean(dim=1)
        return pixels.permute(0, 3, 1, 2).contiguous()

    def _embed_images(
        self, patches: torch.Tensor, grid: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
        m, p = self.merge_size, self.patch_size
        pixels, positions, heights, widths = [], [], [], []
        for image, (_, height, width) in zip(
            patches.split([h * w for _, h, w in grid]), grid
        ):
            native = image.reshape(height, width, 3, m, p, m, p)
            native = native.permute(0, 3, 1, 5, 4, 6, 2).reshape(-1, p * p * 3)
            pixels.extend((native.new_zeros((1, native.shape[1])), native))
            rows = torch.arange(height * width * m * m, device=patches.device)
            coordinates = torch.stack((rows // (width * m), rows % (width * m)), -1)
            positions.extend((coordinates.new_zeros((1, 2)), coordinates))
            heights.append(rows.new_full((rows.numel() + 1,), height * m))
            widths.append(rows.new_full((rows.numel() + 1,), width * m))
        pixels, positions = torch.cat(pixels), torch.cat(positions)
        lengths = [h * w * m * m + 1 for _, h, w in grid]
        cu_seqlens = host_to_device(
            [0, *accumulate(lengths)], dtype=torch.int32, device=patches.device
        )
        embeddings = self.backbone.embeddings
        grid_size = self.config.posemb_grid_size
        hidden = embeddings(pixels)
        hidden = hidden + _position_embeddings(
            embeddings.position_embedding.reshape(grid_size, grid_size, -1),
            positions[:, 0].contiguous(),
            positions[:, 1].contiguous(),
            torch.cat(heights),
            torch.cat(widths),
        )
        cls_rows = cu_seqlens[:-1].long()
        hidden[cls_rows] = (
            embeddings.class_embedding + embeddings.cls_position_embedding
        )
        positions = (positions + 1).to(torch.int32)
        positions.index_fill_(0, cls_rows, 0)
        return self.backbone.ln_pre(hidden), positions, lengths, cu_seqlens

    def _pool(
        self, hidden: torch.Tensor, grid: list[list[int]], lengths: list[int]
    ) -> torch.Tensor:
        m, width = self.merge_size, hidden.shape[-1]
        return torch.cat(
            [
                image[1:]
                .reshape(h, m, w, m, width)
                .permute(0, 2, 1, 3, 4)
                .reshape(h * w, m * m * width)
                for image, (_, h, w) in zip(hidden.split(lengths), grid)
            ]
        )

    def forward(
        self, vision_patches_bthwc: torch.Tensor, vision_grid_thw: torch.Tensor
    ) -> torch.Tensor:
        grid = vision_grid_thw.tolist()
        patches = self._preprocess_patches(vision_patches_bthwc)
        hidden, positions, lengths, cu_seqlens = self._embed_images(patches, grid)
        for layer in self.backbone.layers:
            hidden = layer(hidden, positions, lengths, cu_seqlens)
        hidden = self._pool(hidden, grid, lengths)
        return self.proj_out(F.gelu(self.proj_in(hidden), approximate="none"))
