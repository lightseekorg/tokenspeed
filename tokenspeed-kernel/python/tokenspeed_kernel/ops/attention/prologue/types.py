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

"""Typed holders of the prologue's inputs and outputs, and the trait values
solutions declare."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

BOOLS = frozenset({True, False})
ROPE_STYLES = frozenset({"none", "neox", "gptj"})
# Head widths the CUDA embedding.rope kernel serves.
CUDA_ROPE_HEAD_DIMS = frozenset({64, 128, 256, 512})


class RopeStyle(Enum):
    """Pairing of rotated channels."""

    NEOX = "neox"
    GPTJ = "gptj"


class KVCacheFormat(Enum):
    """How rows are stored in the KV cache."""

    NATIVE = "native"
    FP8 = "fp8"
    MXFP8 = "mxfp8"
    FP8_PER_TOKEN_HEAD = "fp8_per_token_head"


@dataclass(frozen=True)
class HeadNorm:
    """Per-head RMSNorm of query and key heads: ``x * rsqrt(mean(x^2) + eps)
    * (weight + weight_offset)``, the multiplier formed in fp32.

    Attributes:
        q_weight: Stored query weight ``[head_dim]``.
        k_weight: Stored key weight ``[head_dim]``.
        weight_offset: 1.0 for Gemma-style ``1 + w`` weights, else 0.0.
        eps: Epsilon added to the mean square.
    """

    q_weight: torch.Tensor
    k_weight: torch.Tensor
    weight_offset: float
    eps: float


@dataclass(frozen=True)
class MRope:
    """Multimodal RoPE: which position row drives each rotary pair.

    Attributes:
        section: Rotary pairs taken from the T, H and W rows; sums to
            ``rotary_dim // 2``.
        interleaved: Rows alternate per pair instead of occupying contiguous
            sections.
    """

    section: tuple[int, ...]
    interleaved: bool


@dataclass(frozen=True)
class Rotary:
    """Rotary embedding of the leading ``rotary_dim`` channels of each head.

    Attributes:
        cos_sin_cache: Contiguous fp32 ``[max_position, rotary_dim]`` as
            concat(cos, sin).
        positions: ``[num_tokens]``, or ``[3, num_tokens]`` T/H/W rows with
            ``mrope``; 1-D positions rotate every pair by the same row.
        style: Channel pairing.
        mrope: Multimodal sections, or ``None`` for ordinary RoPE.
    """

    cos_sin_cache: torch.Tensor
    positions: torch.Tensor
    style: RopeStyle
    mrope: MRope | None

    @property
    def rotary_dim(self) -> int:
        return int(self.cos_sin_cache.shape[-1])


@dataclass(frozen=True)
class MXFP8Scales:
    """UE8M0 scale planes of an MXFP8 cache, one scale per 32 channels.

    Attributes:
        k: Key scale plane in the interleaved paged layout.
        v: Value scale plane in the same layout.
        page_tokens: Tokens one scale page spans.
    """

    k: torch.Tensor
    v: torch.Tensor
    page_tokens: int


@dataclass(frozen=True)
class HeadKVCache:
    """One GQA layer's KV cache destination.

    Attributes:
        k_cache: ``[slots, num_kv_heads, head_dim]``, FP8 e4m3 or native rows.
        v_cache: ``[slots, num_kv_heads, head_dim]`` in the same dtype.
        scales: The scale planes of an MXFP8 cache, else ``None``.
        slots: Dense 1-D destination slot of each written row; rows past its
            length (graph padding) are not written.
    """

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    scales: MXFP8Scales | None
    slots: torch.Tensor

    @property
    def format(self) -> KVCacheFormat:
        if self.scales is not None:
            return KVCacheFormat.MXFP8
        if self.k_cache.dtype == torch.float8_e4m3fn:
            return KVCacheFormat.FP8
        return KVCacheFormat.NATIVE


@dataclass(frozen=True)
class PerTokenHeadPlanes:
    """The three planes of an FP8_PER_TOKEN_HEAD latent cache.

    Attributes:
        latent: FP8 latent bytes ``[slots, 1, kv_lora_rank]``.
        scale: fp32 per-token scale ``[slots, 1, 1]``.
        rope: RoPE rows divided by the scale ``[slots, 1, rope_dim]``.
    """

    latent: torch.Tensor
    scale: torch.Tensor
    rope: torch.Tensor


@dataclass(frozen=True)
class LatentKVCache:
    """One MLA layer's latent cache destination.

    Attributes:
        kv_cache: ``[slots, 1, kv_lora_rank + rope_dim]`` FP8 e4m3 or native
            rows, or the planes of an FP8_PER_TOKEN_HEAD cache.
        sanitize: Replace NaN/Inf with finite values before storing.
        slots: Dense 1-D destination slot of each written row.
        write_mask: True for each row to store, or None to store every row;
            a skipped row's slot is still a valid address.
    """

    kv_cache: torch.Tensor | PerTokenHeadPlanes
    sanitize: bool
    slots: torch.Tensor
    write_mask: torch.Tensor | None

    @property
    def format(self) -> KVCacheFormat:
        if isinstance(self.kv_cache, PerTokenHeadPlanes):
            return KVCacheFormat.FP8_PER_TOKEN_HEAD
        if self.kv_cache.dtype == torch.float8_e4m3fn:
            return KVCacheFormat.FP8
        return KVCacheFormat.NATIVE


@dataclass(frozen=True)
class MLAExpandedKV:
    """Per-head keys and values up-projected from the latent, for MLA
    attention that does not absorb the up-projection into the query.

    Attributes:
        k_nope: Non-RoPE key part ``[num_tokens, num_heads, q_nope_dim]``.
        value: ``[num_tokens, num_heads, v_head_dim]``.
    """

    k_nope: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class GQAPrologueOutput:
    """Inputs for GQA core attention.

    Attributes:
        q: Rotated query ``[num_tokens, num_q_heads * head_dim]``.
        k: Rotated key rows when requested, else ``None``.
        v: Value rows when requested, else ``None``.
    """

    q: torch.Tensor
    k: torch.Tensor | None
    v: torch.Tensor | None


@dataclass(frozen=True)
class MLAPrologueOutput:
    """Inputs for MLA core attention.

    Attributes:
        query: ``[num_tokens, num_heads, q_nope_dim + rope_dim]``, FP8 for an
            FP8 cache (not per-token-head planes), else in the activation dtype;
            a fresh tensor for expanded attention or an FP8 cache, else the
            given query.
        key: Per-head ``[num_tokens, num_heads, q_nope_dim + rope_dim]`` keys
            in the returned query's dtype for expanded attention, else ``None``.
        value: Per-head values in the returned query's dtype for expanded attention,
            else ``None``.
    """

    query: torch.Tensor
    key: torch.Tensor | None
    value: torch.Tensor | None
