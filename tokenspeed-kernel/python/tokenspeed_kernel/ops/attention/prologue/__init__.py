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

"""Attention prologue: everything between the QKV projections and core attention.

A prologue takes a layer's projected query and key/value rows and leaves the
query ready for the attention kernel and the key/value rows in the KV cache:
optional per-head QK RMSNorm, optional rotary embedding (plain or multimodal),
KV quantization for the cache format, and the KV cache write. Models describe
the steps once; which kernels run them is a dispatch decision made here, so a
fused kernel covering several steps can replace the step-by-step path without
any model knowing.

Two layouts share one vocabulary:

* :func:`gqa_attention_prologue` -- grouped-query (and multi-head) attention
  with separate K and V cache planes.
* :func:`mla_attention_prologue` -- multi-head latent attention with one
  latent-plus-RoPE cache plane.

docs/design/attention-prologue.md states the numerics contract. Inputs may be
overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import torch
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import (
    SelectedKernel,
    select_kernel,
    spec_matches_shape_traits,
    spec_matches_traits,
)
from tokenspeed_kernel.signature import (
    FormatSignature,
    dense_tensor_format,
    format_signature,
)

_BOOLS = frozenset({True, False})
_ROPE_STYLES = frozenset({"none", "neox", "gptj"})
# Head widths the CUDA embedding.rope kernel serves.
_CUDA_ROPE_HEAD_DIMS = frozenset({64, 128, 256, 512})


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
    """

    kv_cache: torch.Tensor | PerTokenHeadPlanes
    sanitize: bool
    slots: torch.Tensor

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
            FP8 cache (not per-token-head planes), else in the activation dtype.
        key: Per-head ``[num_tokens, num_heads, q_nope_dim + rope_dim]`` keys
            in the returned query's dtype for expanded attention, else ``None``.
        value: Per-head values in the returned query's dtype for expanded attention,
            else ``None``.
    """

    query: torch.Tensor
    key: torch.Tensor | None
    value: torch.Tensor | None


def _strides(x: torch.Tensor) -> list[int]:
    """The strides that place elements: those of dimensions longer than one."""
    return [s for n, s in zip(x.shape, x.stride()) if n > 1]


def _misaligned(x: torch.Tensor) -> bool:
    """A row or head of ``x`` starts off a 16-byte boundary, which the CUDA kernels read in vectors."""
    return bool(x.numel()) and (
        bool(x.data_ptr() % 16)
        or any(
            s * x.element_size() % 16
            for n, s in zip(x.shape[:-1], x.stride()[:-1])
            if n > 1
        )
    )


def _overlapping(x: torch.Tensor) -> bool:
    """A stride of ``x`` falls short of the span of the dimensions below it."""
    span = 1
    for size, stride in sorted(
        ((n, s) for n, s in zip(x.shape, x.stride()) if n > 1), key=lambda d: d[1]
    ):
        if stride < span:
            return True
        span += (size - 1) * stride
    return False


def _check_slots(cache: HeadKVCache | LatentKVCache, num_tokens: int) -> None:
    slots = cache.slots
    if (
        slots.dim() != 1
        or slots.stride(0) != 1
        or slots.numel() > num_tokens
        or slots.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            f"cache slots {tuple(slots.shape)} are not a dense vector (int32 or int64) "
            f"of at most {num_tokens}"
        )


def _check_rotary(rotary: Rotary, num_tokens: int) -> None:
    positions = rotary.positions
    if (
        positions.ndim not in (1, 2)
        or positions.shape[-1] != num_tokens
        or positions.stride(-1) != 1
        or positions.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            f"positions {tuple(positions.shape)} are not {num_tokens} dense rows "
            f"of int32 or int64"
        )
    table = rotary.cos_sin_cache
    if table.dtype != torch.float32 or table.dim() != 2 or not table.is_contiguous():
        raise ValueError("the cos/sin cache is dense fp32 rows")
    if rotary.rotary_dim % 2:
        raise ValueError(f"rotary width {rotary.rotary_dim} is odd")
    if positions.ndim == 2 and (
        rotary.mrope is None
        or positions.shape[0] != 3
        or len(rotary.mrope.section) != 3
        or min(rotary.mrope.section) < 0
        or sum(rotary.mrope.section) != rotary.rotary_dim // 2
    ):
        raise ValueError(
            "2-D positions are T/H/W rows whose M-RoPE sections split the rotary pairs"
        )


def _check_mxfp8_scales(cache: HeadKVCache, num_kv_heads: int, head_dim: int) -> None:
    if cache.k_cache.dtype != torch.float8_e4m3fn or head_dim != 128:
        raise ValueError("MXFP8 caches store 128-wide FP8 heads")
    page_tokens = cache.scales.page_tokens
    if page_tokens <= 0 or page_tokens % 128:
        raise ValueError(
            f"MXFP8 scale pages span a positive multiple of 128 tokens, not {page_tokens}"
        )
    pages = -(-cache.k_cache.shape[0] // page_tokens)
    scales_needed = pages * page_tokens * num_kv_heads * head_dim // 32
    if any(
        plane.dtype != torch.float8_e8m0fnu
        or plane.storage_offset() % 4
        or plane.numel() % 4
        or not plane.is_contiguous()
        or plane.numel() < scales_needed
        for plane in (cache.scales.k, cache.scales.v)
    ):
        raise ValueError(
            "MXFP8 scale planes are dense e8m0, one scale per 32 channels of every row"
        )


def _check_gqa_request(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    norm: HeadNorm | None,
    rotary: Rotary | None,
    cache: HeadKVCache,
) -> None:
    """Reject a GQA request any solution would misread; metadata only."""
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"q is {q.dtype}; the prologue takes fp16 or bf16")
    if q.dim() not in (2, 3):
        raise ValueError(
            f"q {tuple(q.shape)} is not [tokens, heads * head_dim] or [tokens, heads, head_dim]"
        )
    num_tokens = q.shape[0]
    if cache.k_cache.dim() != 3:
        raise ValueError(
            f"key cache {tuple(cache.k_cache.shape)} is not [slots, heads, head_dim] rows"
        )
    num_kv_heads, head_dim = cache.k_cache.shape[1:]
    if (
        cache.v_cache.shape != cache.k_cache.shape
        or cache.v_cache.dtype != cache.k_cache.dtype
        or any(
            x.stride(-1) != 1 or (num_kv_heads > 1 and x.stride(1) != head_dim)
            for x in (cache.k_cache, cache.v_cache)
        )
    ):
        raise ValueError(
            "key and value caches must be dense rows of one geometry and dtype"
        )
    if (
        q.stride(-1) != 1
        or q.shape[1:].numel() % head_dim
        or (q.dim() == 3 and q.shape[2] != head_dim)
    ):
        raise ValueError(f"q {tuple(q.shape)} is not dense {head_dim}-wide heads")
    if any(
        x.dim() != 2 or x.shape[1] != num_kv_heads * head_dim or x.stride(-1) != 1
        for x in (k, v)
    ):
        raise ValueError(
            f"k {tuple(k.shape)} / v {tuple(v.shape)} are not dense rows of the cache heads"
        )
    if not k.shape[0] == v.shape[0] == num_tokens:
        raise ValueError("q, k and v must have the same number of rows")
    if not k.dtype == v.dtype == q.dtype:
        raise ValueError("q, k and v must share a dtype")
    if any(_overlapping(x) for x in (q, k, v, cache.k_cache, cache.v_cache)):
        raise ValueError("q, k, v and the caches must not share addresses")
    if any(_misaligned(x) for x in (q, k, v, cache.k_cache, cache.v_cache)):
        raise ValueError("q, k, v and the caches must start rows on 16-byte boundaries")
    if norm is not None and any(
        w.shape != (head_dim,) or w.stride(-1) != 1
        for w in (norm.q_weight, norm.k_weight)
    ):
        raise ValueError(f"norm weights are not dense [{head_dim}]")
    if rotary is not None:
        _check_rotary(rotary, num_tokens)
        if rotary.rotary_dim > head_dim:
            raise ValueError(f"rotary width {rotary.rotary_dim} exceeds the head")
    _check_slots(cache, num_tokens)
    if cache.scales is not None:
        _check_mxfp8_scales(cache, num_kv_heads, head_dim)
    if cache.format is KVCacheFormat.NATIVE and cache.k_cache.dtype not in (
        q.dtype,
        torch.bfloat16,
    ):
        raise ValueError(
            f"a native cache holds {q.dtype} or bf16 rows, not {cache.k_cache.dtype}"
        )


def _check_mla_request(
    query: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    expanded: MLAExpandedKV | None,
    rotary: Rotary | None,
    cache: LatentKVCache,
) -> None:
    """Reject an MLA request any solution would misread; metadata only."""
    if query.dim() != 3:
        raise ValueError(f"query {tuple(query.shape)} is not [tokens, heads, channels]")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"query is {query.dtype}; the prologue takes fp16 or bf16")
    if q_pe.dim() != 3 or latent_cache.dim() != 2:
        raise ValueError(
            f"q_pe {tuple(q_pe.shape)} and latent_cache {tuple(latent_cache.shape)} are "
            f"not [tokens, heads, rope] and [tokens, rank + rope]"
        )
    num_tokens, num_heads, width = query.shape
    rope_dim = q_pe.shape[-1]
    kv_lora_rank = latent_cache.shape[-1] - rope_dim
    if latent_cache.shape[0] != num_tokens or kv_lora_rank <= 0:
        raise ValueError(
            f"latent_cache {tuple(latent_cache.shape)} is not [{num_tokens}, rank + {rope_dim}]"
        )
    if q_pe.shape != (num_tokens, num_heads, rope_dim):
        raise ValueError(
            f"q_pe {tuple(q_pe.shape)} does not match query {tuple(query.shape)}"
        )
    if not q_pe.dtype == latent_cache.dtype == query.dtype:
        raise ValueError("query, q_pe and latent_cache must share a dtype")
    if any(x.stride(-1) != 1 for x in (query, q_pe, latent_cache)):
        raise ValueError("query, q_pe and latent_cache must be dense channels")
    if any(_overlapping(x) for x in (query, q_pe, latent_cache)):
        raise ValueError("query, q_pe and latent_cache must not share addresses")
    if any(_misaligned(x) for x in (query, q_pe, latent_cache)):
        raise ValueError(
            "query, q_pe and latent_cache must start rows on 16-byte boundaries"
        )
    rope_view = query[..., width - rope_dim :]
    if (
        q_pe.numel()
        and q_pe.data_ptr() == rope_view.data_ptr()
        and _strides(q_pe) != _strides(rope_view)
    ):
        raise ValueError("q_pe at the query's RoPE channels must be that view")
    if rope_dim and rope_dim not in _CUDA_ROPE_HEAD_DIMS:
        raise ValueError(
            f"MLA RoPE is 64, 128, 256 or 512 channels wide, or absent, not {rope_dim}"
        )
    _check_slots(cache, num_tokens)
    if cache.format is KVCacheFormat.FP8_PER_TOKEN_HEAD:
        planes = cache.kv_cache
        rows = (
            (planes.latent, kv_lora_rank),
            (planes.scale, 1),
            (planes.rope, rope_dim),
        )
        malformed = (
            f"per-token-head planes are not dense [slots, 1, ·] of {kv_lora_rank} "
            f"latent, 1 scale and {rope_dim} RoPE channels"
        )
    else:
        rows = ((cache.kv_cache, kv_lora_rank + rope_dim),)
        malformed = (
            f"latent cache rows are not dense [slots, 1, {kv_lora_rank} + {rope_dim}]"
        )
    if any(
        p.dim() != 3
        or p.shape[1] != 1
        or p.shape[-1] != channels
        or p.stride(-1) != 1
        or _overlapping(p)
        for p, channels in rows
    ):
        raise ValueError(malformed)
    if cache.format is KVCacheFormat.FP8_PER_TOKEN_HEAD and (
        planes.scale.dtype != torch.float32
        or not planes.latent.shape[0] == planes.scale.shape[0] == planes.rope.shape[0]
    ):
        raise ValueError(
            "per-token-head planes share one row count and hold fp32 scales"
        )
    if cache.format is KVCacheFormat.NATIVE and cache.kv_cache.dtype not in (
        query.dtype,
        torch.bfloat16,
    ):
        raise ValueError(
            f"a native cache holds {query.dtype} or bf16 rows, not {cache.kv_cache.dtype}"
        )
    if rotary is not None:
        if rotary.mrope is not None:
            raise ValueError("MLA does not take multimodal RoPE")
        _check_rotary(rotary, num_tokens)
        if rotary.rotary_dim != rope_dim:
            raise ValueError(
                f"rotary width {rotary.rotary_dim} is not the {rope_dim} RoPE channels"
            )
    if expanded is None and width - rope_dim != kv_lora_rank:
        raise ValueError("an absorbed query's non-RoPE part is kv_lora_rank wide")
    if expanded is not None and (
        expanded.k_nope.shape != (num_tokens, num_heads, width - rope_dim)
        or expanded.value.dim() != 3
        or expanded.value.shape[:2] != (num_tokens, num_heads)
        or not expanded.k_nope.dtype == expanded.value.dtype == query.dtype
        or expanded.k_nope.stride(-1) != 1
        or expanded.value.stride(-1) != 1
        or _overlapping(expanded.k_nope)
        or _overlapping(expanded.value)
    ):
        raise ValueError(
            f"expanded k_nope {tuple(expanded.k_nope.shape)} / value "
            f"{tuple(expanded.value.shape)} are not dense heads of query {tuple(query.shape)}"
        )


@lru_cache(maxsize=4096)
def _serves(
    kernel_name: str, mode: str, traits: tuple[tuple[str, object], ...]
) -> bool:
    spec = KernelRegistry.get().get_by_name(kernel_name)
    return (
        (spec.family, spec.mode) == ("attention", mode)
        and spec.capability.satisfied_by(current_platform())
        and spec_matches_traits(spec, dict(traits))
        and spec_matches_shape_traits(spec, dict(traits))
    )


def _select(
    mode: str,
    signature: FormatSignature,
    traits: dict[str, object],
    solution: str | None,
    override: str | None,
) -> SelectedKernel:
    """Select the kernel for a request; an override may name only a kernel this
    platform runs whose traits cover the request, since solutions drop inputs
    their traits exclude."""
    kernel = select_kernel(
        "attention",
        mode,
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    if not _serves(kernel.name, mode, tuple(traits.items())):
        raise ValueError(f"{kernel.name} does not serve attention.{mode} with {traits}")
    return kernel


def gqa_attention_prologue(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    norm: HeadNorm | None,
    rotary: Rotary | None,
    cache: HeadKVCache,
    return_kv: bool,
    solution: str | None,
    override: str | None,
) -> GQAPrologueOutput:
    """Prepare one GQA layer's query and write its key/value rows to the cache.

    Head counts and widths are read from ``cache``.

    Args:
        q: Query ``[num_tokens, num_q_heads * head_dim]``, or ``[num_tokens,
            num_q_heads, head_dim]``; rows and heads may be strided (a view of
            the packed QKV projection) but each head is contiguous.
        k: Key ``[num_tokens, num_kv_heads * head_dim]``.
        v: Value ``[num_tokens, num_kv_heads * head_dim]``.
        norm: Per-head RMSNorm, or ``None``.
        rotary: Rotary embedding, or ``None`` for NoPE.
        cache: KV cache destination.
        return_kv: Also return the rotated key and the value rows, for
            attention kernels that read them instead of the cache.
        solution: Optional registered solution to select.
        override: Optional exact kernel name; it must serve the request.

    Returns:
        The query and optional key/value rows for core attention.
    """
    _check_gqa_request(q, k, v, norm, rotary, cache)
    num_tokens = q.shape[0]
    num_kv_heads, head_dim = cache.k_cache.shape[1:]
    num_q_heads = q.shape[1:].numel() // head_dim

    traits = {
        "head_dim": head_dim,
        "token_heads": num_tokens * num_q_heads,
        "full_write": cache.slots.numel() == num_tokens,
        "has_norm": norm is not None,
        "kv_format": cache.format.value,
        "kv_convert": cache.format is KVCacheFormat.NATIVE
        and cache.k_cache.dtype is not q.dtype,
        "mrope": rotary is not None and rotary.positions.ndim == 2,
        "partial_rotary": rotary is not None and rotary.rotary_dim != head_dim,
        "return_kv": return_kv,
        "rope_style": "none" if rotary is None else rotary.style.value,
    }
    kernel = _select(
        "gqa_prologue",
        format_signature(q=dense_tensor_format(q.dtype)),
        traits,
        solution,
        override,
    )
    shape_params = {
        "num_tokens": num_tokens,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }
    ShapeCapture.get().record(
        "attention", "gqa_prologue", kernel.name, q.dtype, shape_params
    )
    with kernel_scope(
        "attention", "gqa_prologue", q.dtype, kernel_name=kernel.name, **shape_params
    ):
        return kernel(
            q=q,
            k=k,
            v=v,
            norm=norm,
            rotary=rotary,
            cache=cache,
            return_kv=return_kv,
            enable_pdl=pdl_enabled(),
        )


def qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    head_dim: int,
    norm: HeadNorm | None,
    rotary: Rotary | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize and rotate query and key heads without writing a cache.

    For keys that are not attention K/V, such as a sparse-attention indexer's;
    the GQA prologue's kernels and numerics, with no cache rows to write.

    Args:
        q: Query ``[num_tokens, q_heads * head_dim]``.
        k: Key ``[num_tokens, k_heads * head_dim]``.
        head_dim: Width of one head.
        norm: Per-head RMSNorm, or ``None``.
        rotary: Rotary embedding, or ``None`` for NoPE.

    Returns:
        The query and key, each ``[num_tokens, heads * head_dim]``.
    """
    no_rows = k.new_empty(0, k.shape[1:].numel() // head_dim, head_dim)
    out = gqa_attention_prologue(
        q,
        k,
        k,
        norm=norm,
        rotary=rotary,
        cache=HeadKVCache(
            k_cache=no_rows,
            v_cache=no_rows,
            scales=None,
            slots=k.new_empty(0, dtype=torch.int64),
        ),
        return_kv=True,
        solution=None,
        override=None,
    )
    return out.q, out.k


def mla_attention_prologue(
    query: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    *,
    expanded: MLAExpandedKV | None,
    rotary: Rotary | None,
    cache: LatentKVCache,
    solution: str | None,
    override: str | None,
) -> MLAPrologueOutput:
    """Prepare one MLA layer's attention inputs and write its latent rows.

    Args:
        query: ``[num_tokens, num_heads, q_nope_dim + rope_dim]`` buffer whose
            leading channels already hold the query's non-RoPE part: the
            absorbed latent query (``q_nope_dim == kv_lora_rank``) or a
            per-head query that attends expanded keys. Its RoPE channels
            receive the rotated ``q_pe`` unless the cache is FP8 (not planes).
        q_pe: Unrotated query RoPE part ``[num_tokens, num_heads, rope_dim]``;
            it may alias ``query``'s RoPE channels.
        latent_cache: ``[num_tokens, kv_lora_rank + rope_dim]`` normalized
            latent followed by the unrotated key RoPE part, which may be
            rotated in place.
        expanded: Per-head keys and values when attention does not absorb
            the latent up-projection; ``None`` for absorbed attention, which
            reads the latent cache.
        rotary: Rotary embedding, or ``None`` for NoPE.
        cache: Latent cache destination.
        solution: Optional registered solution to select.
        override: Optional exact kernel name; it must serve the request.

    Returns:
        The query, plus the per-head keys and values of expanded attention:
        FP8 e4m3 for an FP8 cache (not per-token-head planes), else in the query dtype.
    """
    _check_mla_request(query, q_pe, latent_cache, expanded, rotary, cache)
    num_tokens, num_heads, width = query.shape
    rope_dim = q_pe.shape[-1]
    kv_lora_rank = latent_cache.shape[-1] - rope_dim

    traits = {
        "token_heads": num_tokens * num_heads,
        "expanded": expanded is not None,
        "full_write": cache.slots.numel() == num_tokens,
        "kv_format": cache.format.value,
        "kv_convert": cache.format is KVCacheFormat.NATIVE
        and cache.kv_cache.dtype is not query.dtype,
        "rope_style": "none" if rotary is None else rotary.style.value,
        "sanitize": cache.sanitize,
    }
    kernel = _select(
        "mla_prologue",
        format_signature(query=dense_tensor_format(query.dtype)),
        traits,
        solution,
        override,
    )
    shape_params = {
        "num_tokens": num_tokens,
        "num_heads": num_heads,
        "q_nope_dim": width - rope_dim,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": rope_dim,
    }
    ShapeCapture.get().record(
        "attention", "mla_prologue", kernel.name, query.dtype, shape_params
    )
    with kernel_scope(
        "attention",
        "mla_prologue",
        query.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            query=query,
            q_pe=q_pe,
            latent_cache=latent_cache,
            expanded=expanded,
            rotary=rotary,
            cache=cache,
            enable_pdl=pdl_enabled(),
        )


__all__ = [
    "GQAPrologueOutput",
    "HeadKVCache",
    "HeadNorm",
    "KVCacheFormat",
    "LatentKVCache",
    "MLAExpandedKV",
    "MLAPrologueOutput",
    "MRope",
    "MXFP8Scales",
    "PerTokenHeadPlanes",
    "RopeStyle",
    "Rotary",
    "gqa_attention_prologue",
    "mla_attention_prologue",
    "qk_norm_rope",
]


# Backend registration (side-effect imports)
import tokenspeed_kernel.ops.attention.prologue.composite  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.prologue.fused_rope  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.prologue.triton  # noqa: E402,F401
