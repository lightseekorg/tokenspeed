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

"""What the prologue entries accept: the request checks run before selecting a
kernel."""

from __future__ import annotations

from functools import lru_cache

import torch
from tokenspeed_kernel.ops.attention.prologue.types import (
    CUDA_ROPE_HEAD_DIMS,
    HeadKVCache,
    HeadNorm,
    KVCacheFormat,
    LatentKVCache,
    MLAExpandedKV,
    PerTokenHeadPlanes,
    Rotary,
)


def _strides(x: torch.Tensor) -> list[int]:
    """The strides that place elements: those of dimensions longer than one."""
    return [s for n, s in zip(x.shape, x.stride()) if n > 1]


@lru_cache(maxsize=4096)
def _misaligned_strides(
    shape: torch.Size, stride: tuple[int, ...], element_size: int
) -> bool:
    return any(s * element_size % 16 for n, s in zip(shape[:-1], stride[:-1]) if n > 1)


def _misaligned(x: torch.Tensor) -> bool:
    """A row or head of ``x`` starts off a 16-byte boundary, which the CUDA kernels read in vectors."""
    return bool(x.numel()) and (
        bool(x.data_ptr() % 16)
        or _misaligned_strides(x.shape, x.stride(), x.element_size())
    )


@lru_cache(maxsize=4096)
def _overlapping_layout(shape: torch.Size, stride: tuple[int, ...]) -> bool:
    span = 1
    for size, step in sorted(
        ((n, s) for n, s in zip(shape, stride) if n > 1), key=lambda d: d[1]
    ):
        if step < span:
            return True
        span += (size - 1) * step
    return False


def _overlapping(x: torch.Tensor) -> bool:
    """A stride of ``x`` falls short of the span of the dimensions below it."""
    return _overlapping_layout(x.shape, x.stride())


def _layout(x: torch.Tensor) -> tuple:
    return (x.shape, x.stride(), x.dtype, x.data_ptr() % 16, x.storage_offset() % 4)


def _rotary_key(rotary: Rotary | None) -> tuple | None:
    if rotary is None:
        return None
    return (
        _layout(rotary.cos_sin_cache),
        _layout(rotary.positions),
        rotary.style,
        rotary.mrope,
    )


def _cache_key(cache: HeadKVCache | LatentKVCache) -> tuple:
    if isinstance(cache, HeadKVCache):
        scales = cache.scales
        planes = (
            None
            if scales is None
            else (_layout(scales.k), _layout(scales.v), scales.page_tokens)
        )
        return (
            _layout(cache.k_cache),
            _layout(cache.v_cache),
            planes,
            _layout(cache.slots),
        )
    kv = cache.kv_cache
    kv_key = (
        _layout(kv)
        if isinstance(kv, torch.Tensor)
        else (_layout(kv.latent), _layout(kv.scale), _layout(kv.rope))
    )
    return (kv_key, cache.sanitize, _layout(cache.slots))


_checked: set[tuple] = set()


def _once(key: tuple, check, *args) -> None:
    """Run a metadata check once per layout: requests repeat layouts, so the checks need not."""
    if key in _checked:
        return
    check(*args)
    if len(_checked) >= 4096:
        _checked.clear()
    _checked.add(key)


def check_gqa_request(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    norm: HeadNorm | None,
    rotary: Rotary | None,
    cache: HeadKVCache,
) -> None:
    """Reject a GQA request any solution would misread; metadata only."""
    norm_key = (
        None if norm is None else (_layout(norm.q_weight), _layout(norm.k_weight))
    )
    key = (
        "gqa",
        _layout(q),
        _layout(k),
        _layout(v),
        norm_key,
        _rotary_key(rotary),
        _cache_key(cache),
    )
    _once(key, _check_gqa_request, q, k, v, norm, rotary, cache)


def check_latent_write(
    latent_cache: torch.Tensor, rotary: Rotary | None, cache: LatentKVCache
) -> None:
    """Reject a latent write any store would misread; metadata only."""
    key = ("latent", _layout(latent_cache), _rotary_key(rotary), _cache_key(cache))
    _once(key, _check_latent_write, latent_cache, rotary, cache)


def _check_latent_write(
    latent_cache: torch.Tensor, rotary: Rotary | None, cache: LatentKVCache
) -> None:
    if latent_cache.dim() != 2 or latent_cache.stride(-1) != 1:
        raise ValueError(
            f"latent_cache {tuple(latent_cache.shape)} is not [tokens, rank + rope] rows"
        )
    if latent_cache.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"latent_cache is {latent_cache.dtype}; the prologue takes fp16 or bf16"
        )
    num_tokens, width = latent_cache.shape
    if rotary is not None:
        _check_rotary(rotary, num_tokens)
        if width <= rotary.rotary_dim:
            raise ValueError(
                f"latent_cache width {width} leaves no latent beside {rotary.rotary_dim} RoPE channels"
            )
    _check_slots(cache, num_tokens)
    if cache.slots.numel() != num_tokens:
        raise ValueError(f"{cache.slots.numel()} slots for {num_tokens} latent rows")
    kv = cache.kv_cache
    if isinstance(kv, PerTokenHeadPlanes):
        return
    if kv.dim() != 3 or kv.shape[1] != 1 or kv.shape[2] != width or kv.stride(-1) != 1:
        raise ValueError(
            f"latent cache {tuple(kv.shape)} is not [slots, 1, {width}] rows"
        )
    if cache.format is KVCacheFormat.NATIVE and kv.dtype not in (
        latent_cache.dtype,
        torch.bfloat16,
    ):
        raise ValueError(
            f"a native cache holds {latent_cache.dtype} or bf16 rows, not {kv.dtype}"
        )


def check_mla_request(
    query: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    expanded: MLAExpandedKV | None,
    rotary: Rotary | None,
    cache: LatentKVCache,
) -> None:
    """Reject an MLA request any solution would misread; metadata only."""
    expanded_key = (
        None
        if expanded is None
        else (_layout(expanded.k_nope), _layout(expanded.value))
    )
    key = (
        "mla",
        _layout(query),
        _layout(q_pe),
        q_pe.data_ptr() - query.data_ptr(),
        _layout(latent_cache),
        expanded_key,
        _rotary_key(rotary),
        _cache_key(cache),
    )
    _once(key, _check_mla_request, query, q_pe, latent_cache, expanded, rotary, cache)


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
    if rope_dim and rope_dim not in CUDA_ROPE_HEAD_DIMS:
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
