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

"""Official DeepSeek V4 Flash Vision tower and multimodal forward semantics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from tokenspeed.runtime.multimodal.inputs import (
        MultimodalDataItem,
        MultimodalForwardContext,
    )

IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEWLINE, IMAGE_END = range(5)
COMPRESS_PAD_TO = 4


class DeepseekV4VisionMetadataError(ValueError):
    """The transported image geometry does not describe its prompt block."""


class DeepseekV4VisionRowCountError(ValueError):
    """The encoded image block and authored placeholder span have different rows."""


@lru_cache(maxsize=8)
def _cached_vision_cos_sin(
    n_h: int,
    n_w: int,
    dim: int,
    theta: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct canonical FP32 2-D RoPE tables on the execution device."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    hpos = torch.arange(n_h, device=device).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w, device=device).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def get_vision_cos_sin(
    n_h: int,
    n_w: int,
    dim: int,
    theta: float,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical FP32 2-D RoPE tables on ``device``."""
    resolved_device = torch.device("cpu" if device is None else device)
    return _cached_vision_cos_sin(n_h, n_w, dim, theta, str(resolved_device))


def apply_vision_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply the official split-half rotary transform in FP32 arithmetic."""
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class DeepseekV4VisionRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # Parameter storage follows the model loader's default dtype (BF16 for
        # the official checkpoint); normalization arithmetic remains FP32.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = x.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(-1, keepdim=True) + self.eps
        )
        return (self.weight.float() * normalized).to(dtype)


class DeepseekV4PatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.proj = nn.Linear(
            3 * int(config.vision_patch_size) ** 2,
            int(config.vision_dim),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.proj(patches.flatten(1))


class DeepseekV4VisionAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.n_heads = int(config.vision_n_heads)
        vision_dim = int(config.vision_dim)
        if vision_dim % self.n_heads:
            raise ValueError(
                "DeepSeek V4 vision_dim must be divisible by vision_n_heads"
            )
        self.head_dim = vision_dim // self.n_heads
        self.wqkv = nn.Linear(vision_dim, 3 * vision_dim)
        self.wo = nn.Linear(vision_dim, vision_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = x.shape[0]
        q, k, v = (
            part.view(num_tokens, self.n_heads, self.head_dim)
            for part in self.wqkv(x).chunk(3, dim=-1)
        )
        q = apply_vision_rotary(q, cos, sin)
        k = apply_vision_rotary(k, cos, sin)
        # Native rank-3 SDPA is the checkpoint reference contract: heads are
        # the leading dimension and every image uses full bidirectional attention.
        output = F.scaled_dot_product_attention(
            q.transpose(0, 1),
            k.transpose(0, 1),
            v.transpose(0, 1),
        )
        return self.wo(output.transpose(0, 1).reshape(num_tokens, -1))


class DeepseekV4VisionMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        vision_dim = int(config.vision_dim)
        inter_dim = int(config.vision_inter_dim)
        self.w1 = nn.Linear(vision_dim, 2 * inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, vision_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class DeepseekV4VisionBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        vision_dim = int(config.vision_dim)
        self.norm1 = DeepseekV4VisionRMSNorm(vision_dim)
        self.attn = DeepseekV4VisionAttention(config)
        self.norm2 = DeepseekV4VisionRMSNorm(vision_dim)
        self.mlp = DeepseekV4VisionMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class DeepseekV4VisionTransformer(nn.Module):
    """Inference-only official ViT with full bidirectional image attention."""

    def __init__(self, config) -> None:
        super().__init__()
        vision_dim = int(config.vision_dim)
        vision_heads = int(config.vision_n_heads)
        self.rope_dim = vision_dim // vision_heads // 2
        self.rope_theta = float(config.vision_rope_theta)
        self.patch_embed = DeepseekV4PatchEmbed(config)
        self.blocks = nn.ModuleList(
            DeepseekV4VisionBlock(config) for _ in range(int(config.vision_n_layers))
        )
        self.norm = DeepseekV4VisionRMSNorm(vision_dim)

    def forward(
        self,
        patches: torch.Tensor,
        n_h: int,
        n_w: int,
    ) -> torch.Tensor:
        if patches.shape[0] != n_h * n_w:
            raise DeepseekV4VisionMetadataError(
                "DeepSeek V4 vision patch count mismatch: "
                f"patches={patches.shape[0]}, n_vit_h={n_h}, n_vit_w={n_w}"
            )
        hidden_states = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(
            n_h,
            n_w,
            self.rope_dim,
            self.rope_theta,
            device=hidden_states.device,
        )
        for block in self.blocks:
            hidden_states = block(hidden_states, cos, sin)
        return self.norm(hidden_states)


class DeepseekV4VisionAligner(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.downsample_ratio = int(config.vision_downsample_ratio)
        input_dim = int(config.vision_dim) * self.downsample_ratio**2
        hidden_size = int(config.hidden_size)
        self.w1 = nn.Linear(input_dim, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        n_h: int,
        n_w: int,
    ) -> torch.Tensor:
        ratio = self.downsample_ratio
        # The view is HWC, then permuted to CHW before unfold. PyTorch unfold
        # consequently emits each feature row in (channel, kh, kw) order.
        chw = hidden_states.view(n_h, n_w, -1).permute(2, 0, 1)
        chw = F.pad(chw, (0, -n_w % ratio, 0, -n_h % ratio))
        windows = F.unfold(chw.unsqueeze(0), ratio, stride=ratio)
        windows = windows.squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(windows), approximate="none"))


def build_image_block(
    n_llm_h: int,
    n_llm_w: int,
    start_pos: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build official final-order sentinel types and aligner-row permutation."""
    if n_llm_h <= 0 or n_llm_w <= 0 or start_pos < 0:
        raise DeepseekV4VisionMetadataError(
            "DeepSeek V4 image block requires positive geometry and start offset"
        )
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEWLINE]) * n_llm_h
        + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    order = (
        torch.arange(rows * row_len)
        .view(rows // 2, 2, row_len)
        .transpose(1, 2)
        .reshape(-1)
    )
    image_indices = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_indices.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(
        n_llm_h * n_llm_w
    ).view(n_llm_h, n_llm_w)
    permutation = image_indices[order]
    permutation = permutation[permutation >= 0]
    types = torch.cat(
        [
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_START]),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_END]),
        ]
    )
    return types, permutation


def _metadata_scalar(item: MultimodalDataItem, name: str) -> int:
    value = item.model_specific_data.get(name)
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise DeepseekV4VisionMetadataError(
            f"DeepSeek V4 image metadata {name!r} must be a one-element tensor"
        )
    return int(value.reshape(-1)[0].item())


def _placeholder_count(item: MultimodalDataItem) -> int:
    if not item.offsets:
        return 0
    return sum(end - start + 1 for start, end in item.offsets)


def merge_image_block(
    aligned: torch.Tensor,
    types: torch.Tensor,
    permutation: torch.Tensor,
    sentinels: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    placeholder_count: int,
) -> torch.Tensor:
    """Merge sentinels and permuted aligner rows into one scatter-ready block."""
    if aligned.shape[0] != permutation.numel():
        raise DeepseekV4VisionRowCountError(
            "DeepSeek V4 aligned image row count does not match the N-layout "
            f"permutation: aligned={aligned.shape[0]}, image_rows={permutation.numel()}"
        )
    if types.numel() != placeholder_count:
        raise DeepseekV4VisionRowCountError(
            "DeepSeek V4 encoded block row count does not match placeholder-token "
            f"count: encoded={types.numel()}, placeholders={placeholder_count}"
        )
    image_start, image_end, image_newline, image_pad = sentinels
    parameters = torch.stack(
        [image_start, image_pad, image_pad, image_newline, image_end]
    )
    device_types = types.to(device=aligned.device)
    block = parameters[device_types]
    image_slots = device_types == IMAGE
    block[image_slots] = aligned[permutation.to(device=aligned.device)]
    return block


def encode_image_items(
    items: list[MultimodalDataItem],
    vision: DeepseekV4VisionTransformer,
    aligner: DeepseekV4VisionAligner,
    sentinels: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Validate, encode, and merge transported DeepSeek V4 image items."""
    outputs = []
    for item in items:
        if getattr(item.modality, "name", None) != "IMAGE":
            raise DeepseekV4VisionMetadataError(
                "DeepSeek V4 vision encoder received a non-image item"
            )
        if not isinstance(item.feature, torch.Tensor):
            raise DeepseekV4VisionMetadataError(
                "DeepSeek V4 image item has no materialized patch tensor"
            )
        if item.feature.ndim != 4 or item.feature.shape[1] != 3:
            raise DeepseekV4VisionMetadataError(
                "DeepSeek V4 patches must use [tokens, channel, height, width] "
                f"layout; got {tuple(item.feature.shape)}"
            )
        if not item.offsets or len(item.offsets) != 1:
            raise DeepseekV4VisionMetadataError(
                "DeepSeek V4 image items require exactly one authored placeholder span"
            )
        block_start, block_end = item.offsets[0]
        if block_start < 0 or block_end < block_start:
            raise DeepseekV4VisionMetadataError(
                f"DeepSeek V4 image span is invalid: {(block_start, block_end)}"
            )

        n_vit_h = _metadata_scalar(item, "n_vit_h")
        n_vit_w = _metadata_scalar(item, "n_vit_w")
        n_llm_h = _metadata_scalar(item, "n_llm_h")
        n_llm_w = _metadata_scalar(item, "n_llm_w")
        transmitted_pad = _metadata_scalar(item, "dsv4_compress_pad")
        offset_pad = COMPRESS_PAD_TO - 1 - block_start % COMPRESS_PAD_TO
        types, permutation = build_image_block(n_llm_h, n_llm_w, block_start)
        block_len = block_end - block_start + 1
        base_types, _ = build_image_block(n_llm_h, n_llm_w, 3)
        block_pad = block_len - base_types.numel()
        if transmitted_pad != offset_pad or transmitted_pad != block_pad:
            raise DeepseekV4VisionMetadataError(
                "DeepSeek V4 compress_pad mismatch: "
                f"transmitted={transmitted_pad}, offset_derived={offset_pad}, "
                f"block_derived={block_pad}"
            )
        placeholder_count = _placeholder_count(item)
        if types.numel() != placeholder_count:
            raise DeepseekV4VisionRowCountError(
                "DeepSeek V4 encoded block row count does not match placeholder-token "
                f"count: encoded={types.numel()}, placeholders={placeholder_count}"
            )

        projected = vision(item.feature, n_vit_h, n_vit_w)
        aligned = aligner(projected, n_vit_h, n_vit_w)
        outputs.append(
            merge_image_block(
                aligned,
                types,
                permutation,
                sentinels,
                placeholder_count=placeholder_count,
            )
        )
    if not outputs:
        hidden_size = sentinels[0].numel()
        return sentinels[0].new_empty((0, hidden_size))
    return torch.cat(outputs, dim=0)


@dataclass(frozen=True)
class DSV4VisionForward:
    """Semantic image metadata in the current flattened forward's token order."""

    image_mask: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    atomic_spans_in_chunk: list[list[tuple[int, int]]]
    visibility_spans_in_chunk: list[list[tuple[int, int]]]
    atomic_spans: list[list[tuple[int, int]]]
    visibility_spans: list[list[tuple[int, int]]]

    @property
    def intersects_span(self) -> bool:
        return any(self.atomic_spans_in_chunk)


def _request_atomic_spans(
    multimodal_context: MultimodalForwardContext,
) -> list[list[tuple[int, int]]]:
    spans_by_request: list[list[tuple[int, int]]] = []
    for mm_inputs in multimodal_context.mm_inputs:
        spans = []
        if mm_inputs is not None:
            for item in mm_inputs.mm_items:
                if (
                    item is None
                    or getattr(item.modality, "name", None) != "IMAGE"
                    or not item.offsets
                ):
                    continue
                spans.extend((int(start), int(end)) for start, end in item.offsets)
        spans.sort()
        for index, (start, end) in enumerate(spans):
            if start < 0 or end < start:
                raise DeepseekV4VisionMetadataError(
                    f"DeepSeek V4 image span is invalid: {(start, end)}"
                )
            if index and start <= spans[index - 1][1]:
                raise DeepseekV4VisionMetadataError(
                    "DeepSeek V4 image spans must be strictly ordered and non-overlapping"
                )
        spans_by_request.append(spans)
    return spans_by_request


def build_dsv4_vision_forward(
    multimodal_context: MultimodalForwardContext,
    *,
    num_tokens: int,
    device: torch.device | str,
    max_image_tokens: int,
) -> DSV4VisionForward | None:
    """Build image semantics solely from transport offsets and forward geometry."""
    if num_tokens < 0:
        raise ValueError("DeepSeek V4 forward token count must be non-negative")
    atomic_spans = _request_atomic_spans(multimodal_context)
    visibility_spans = [
        [(start + 3 - start % 4, end) for start, end in request_spans]
        for request_spans in atomic_spans
    ]
    atomic_in_chunk: list[list[tuple[int, int]]] = []
    visibility_in_chunk: list[list[tuple[int, int]]] = []
    for request_index, request_spans in enumerate(atomic_spans):
        if request_index >= len(multimodal_context.extend_seq_lens):
            atomic_in_chunk.append([])
            visibility_in_chunk.append([])
            continue
        query_len = int(multimodal_context.extend_seq_lens[request_index])
        query_start = int(multimodal_context.extend_prefix_lens[request_index])
        query_end = query_start + query_len - 1
        intersecting = [
            (start, end)
            for start, end in request_spans
            if query_len > 0 and start <= query_end and end >= query_start
        ]
        atomic_in_chunk.append(intersecting)
        visibility_in_chunk.append(
            [(start + 3 - start % 4, end) for start, end in intersecting]
        )

    if not any(atomic_in_chunk):
        return None

    image_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    left = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    right = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    flat_base = 0
    for request_index, request_atomic in enumerate(atomic_in_chunk):
        if request_index >= len(multimodal_context.extend_seq_lens):
            break
        query_len = int(multimodal_context.extend_seq_lens[request_index])
        query_start = int(multimodal_context.extend_prefix_lens[request_index])
        query_end = query_start + query_len - 1
        for start, end in request_atomic:
            local_start = flat_base + max(start, query_start) - query_start
            local_end = flat_base + min(end, query_end) - query_start
            image_mask[local_start : local_end + 1] = True
        for start, end in visibility_in_chunk[request_index]:
            visible_start = max(start, query_start)
            visible_end = min(end, query_end)
            if visible_start > visible_end:
                continue
            local_start = flat_base + visible_start - query_start
            local_end = flat_base + visible_end - query_start
            positions = torch.arange(
                visible_start,
                visible_end + 1,
                dtype=torch.int32,
                device=device,
            )
            left[local_start : local_end + 1] = (positions - start).clamp(
                max=max_image_tokens - 1
            )
            right[local_start : local_end + 1] = (end - positions).clamp(
                max=max_image_tokens
            )
        flat_base += query_len
    if flat_base > num_tokens:
        raise DeepseekV4VisionMetadataError(
            "DeepSeek V4 extend rows exceed the flattened forward token count: "
            f"extend_tokens={flat_base}, forward_tokens={num_tokens}"
        )
    return DSV4VisionForward(
        image_mask=image_mask,
        left=left,
        right=right,
        atomic_spans_in_chunk=atomic_in_chunk,
        visibility_spans_in_chunk=visibility_in_chunk,
        atomic_spans=atomic_spans,
        visibility_spans=visibility_spans,
    )


__all__ = [
    "COMPRESS_PAD_TO",
    "DSV4VisionForward",
    "DeepseekV4VisionAligner",
    "DeepseekV4VisionMetadataError",
    "DeepseekV4VisionRowCountError",
    "DeepseekV4VisionTransformer",
    "apply_vision_rotary",
    "build_dsv4_vision_forward",
    "build_image_block",
    "encode_image_items",
    "get_vision_cos_sin",
    "merge_image_block",
]
