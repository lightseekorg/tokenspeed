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

import math
from dataclasses import dataclass
from enum import Enum

import torch

BOOLS = frozenset({True, False})
ROPE_STYLES = frozenset({"none", "neox", "gptj"})
# A K/V short convolution's batch: none, decode/verify, or extend.
KV_CONVS = frozenset({"none", "decode", "extend"})
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
    FP8_PER_TENSOR = "fp8_per_tensor"
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
        round_before_rope: Require normalized heads to round to the input
            dtype before rotation. False retains the solution's usual numerics.
    """

    q_weight: torch.Tensor
    k_weight: torch.Tensor
    weight_offset: float
    eps: float
    round_before_rope: bool


@dataclass(frozen=True)
class ConvStream:
    """One channel stream of a short causal convolution over each request's
    recent rows, as ``inkling_ring_sconv`` takes it.

    Attributes:
        weight: Taps ``[channels, W]``; tap ``W - 1`` multiplies the current
            row, or tap 0 with ``dynamic_weights``.
        dynamic_weights: Optional floating ``[num_tokens, heads, W]``
            coefficients, each added in fp32 to the taps of its head's
            channels.
        ring: The stream's ``[slots, R, channels]`` view of the conv ring.
        checkpoint: The ``[pages, W - 1, channels]`` field a published
            window goes to.
    """

    weight: torch.Tensor
    dynamic_weights: torch.Tensor | None
    ring: torch.Tensor
    checkpoint: torch.Tensor


@dataclass(frozen=True)
class ConvBatch:
    """One forward's addressing of the conv rings, as ``inkling_ring_sconv``
    takes it.

    Attributes:
        cu_seqlens, seq_idx, cache_indices, has_initial_state, seq_lens,
            block_table: The ring addressing of ``inkling_ring_sconv``.
        num_extends: Extend requests in the batch; 0 is a decode or verify
            batch of uniform chunks.
        publish_live: Each request's last row writes its chunk-end window to
            the live checkpoint block.
    """

    cu_seqlens: torch.Tensor
    seq_idx: torch.Tensor
    cache_indices: torch.Tensor
    has_initial_state: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    num_extends: int
    publish_live: bool


@dataclass(frozen=True)
class KVShortConv:
    """The short causal convolution of the key and value rows before their
    head steps: each stream's ``(x + conv(window)) * output_scale`` rounded
    once to the row dtype, its ring rows and published window written as
    ``inkling_ring_sconv`` writes them.

    Attributes:
        k, v: The key and value streams.
        batch: Their ring addressing.
        output_scale: Finite multiplier of both streams.
    """

    k: ConvStream
    v: ConvStream
    batch: ConvBatch
    output_scale: float


@dataclass(frozen=True)
class RelativeHeadNorm:
    """Relative attention's head steps (Inkling): optionally the short
    convolution of the key and value rows, then the query and key heads
    L2-clipped, the value heads RMS-normalized, and the clipped relative
    heads projected to per-distance logits, each output rounded once as
    ``inkling_ring_sconv`` and ``rel_mha_head_norms`` compute it.

    Attributes:
        relative: Relative heads ``[num_tokens, num_q_heads, rel_dim]``; rows
            may be strided, each head contiguous.
        projection: Relative projection ``[rel_dim, extent]``.
        q_scale, k_scale, rel_scale: Per-channel clip scales, or ``None``.
        q_threshold, k_threshold, rel_threshold: L2 clip thresholds.
        clip_eps: Epsilon of the L2 clips.
        value_norm_eps: Epsilon of the value RMS norm, or ``None`` to store
            the value heads as given.
        q_output_scale: fp32 ``[num_tokens]`` multiplier of the clipped query
            and relative logits, or ``None``.
        kv_conv: The convolution of the key and value rows, which are then
            its inputs, or ``None``.
    """

    relative: torch.Tensor
    projection: torch.Tensor
    q_scale: torch.Tensor | None
    k_scale: torch.Tensor | None
    rel_scale: torch.Tensor | None
    q_threshold: float
    k_threshold: float
    rel_threshold: float
    clip_eps: float
    value_norm_eps: float | None
    q_output_scale: torch.Tensor | None
    kv_conv: KVShortConv | None


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
class FP8TensorScales:
    """Per-tensor scales of an FP8 cache: its rows hold ``K / k`` and ``V / v``.

    Attributes:
        k: Key scale, positive and finite.
        v: Value scale, positive and finite.
    """

    k: float
    v: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(s) and s > 0 for s in (self.k, self.v)):
            raise ValueError(f"FP8 KV-cache scales must be positive: {self}")


@dataclass(frozen=True)
class HeadKVCache:
    """One GQA layer's KV cache destination.

    Attributes:
        k_cache: ``[slots, heads, dim]`` or ``[pages, rows, heads, dim]``;
            native and unit-scale FP8 writes preserve the field's actual
            page stride.
        v_cache: Same geometry and dtype as ``k_cache``.
        scales: The scale planes of an MXFP8 cache, the per-tensor scales of
            an FP8 one, or ``None`` for a unit-scale cache.
        slots: Dense 1-D destination slot of each written row; rows past its
            length (graph padding) are not written.
    """

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    scales: MXFP8Scales | FP8TensorScales | None
    slots: torch.Tensor

    @property
    def format(self) -> KVCacheFormat:
        if isinstance(self.scales, MXFP8Scales):
            return KVCacheFormat.MXFP8
        if isinstance(self.scales, FP8TensorScales):
            return KVCacheFormat.FP8_PER_TENSOR
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
            rows, a strided [pages, page_size, 1, width] view of those rows,
            or the planes of an FP8_PER_TOKEN_HEAD cache.
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
        rel_logits: ``[num_tokens, num_q_heads, extent]`` relative logits
            under a ``RelativeHeadNorm``, else ``None``.
    """

    q: torch.Tensor
    k: torch.Tensor | None
    v: torch.Tensor | None
    rel_logits: torch.Tensor | None = None


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
        latent: The rotated latent rows ``[num_tokens, kv_lora_rank + rope_dim]``
            in the activation dtype when the prologue ran without a cache
            (``cache=None``) and left the store to :func:`latent_store`;
            ``None`` when the prologue wrote the cache itself.
    """

    query: torch.Tensor
    key: torch.Tensor | None
    value: torch.Tensor | None
    latent: torch.Tensor | None = None
