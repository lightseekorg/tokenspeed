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

"""Shared inputs and the fp64 reference for the attention prologue tests.

``ops/attention/test_attention_prologue.py`` covers the portable solutions and
``nvidia/ops/attention/test_fused_attention_prologue.py`` the NVIDIA ones; both
check "rounded once" against the same fp64 norm and rotation.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.prologue import (
    HeadKVCache,
    HeadNorm,
    KVCacheFormat,
    LatentKVCache,
    MRope,
    RopeStyle,
    Rotary,
    gqa_prologue,
)
from tokenspeed_kernel.platform import current_platform

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
POISON = 0x5A
# (explicit mantissa bits, smallest normal exponent) of each output dtype.
_FLOAT_FORMATS = {torch.float16: (10, -14), BF16: (7, -126), FP8: (3, -6)}


def cos_sin_cache(rotary_dim: int, max_pos: int = 4096) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).cuda()


def poisoned(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    width = shape[-1] * torch.empty((), dtype=dtype).element_size()
    raw = torch.full((*shape[:-1], width), POISON, dtype=torch.uint8, device="cuda")
    return raw.view(dtype)


def gqa_cache(
    slots: int, heads: int, dim: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    return poisoned((slots, heads, dim), dtype), poisoned((slots, heads, dim), dtype)


def qkv(
    tokens: int, hq: int, hkv: int, dim: int, seed: int, dtype: torch.dtype = BF16
) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        tokens, (hq + 2 * hkv) * dim, dtype=dtype, device="cuda", generator=g
    )


def split(qkv: torch.Tensor, hq: int, hkv: int, dim: int):
    return qkv.split([hq * dim, hkv * dim, hkv * dim], dim=-1)


def slots(tokens: int, total: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randperm(total, device="cuda", generator=g)[:tokens].to(torch.int64)


def bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def assert_agree(a: torch.Tensor, b: torch.Tensor) -> None:
    """Byte-equal; on AMD two solutions' fp16 results are allowed one ulp apart."""
    if bytes_equal(a, b):
        return
    assert current_platform().is_amd and a.dtype is torch.float16, "bytes differ"
    differ = (a != b).float().mean().item()
    gap = (a.float() - b.float()).abs().max().item()
    assert torch.allclose(
        a.float(), b.float(), rtol=2**-10, atol=2**-14
    ), f"{differ:.1%} of the fp16 values differ, by up to {gap}"


def head_norm(head_dim: int, weight_offset: float, seed: int) -> HeadNorm:
    """Random weights around what a model of this offset stores."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def weight() -> torch.Tensor:
        w = torch.rand(head_dim, device="cuda", generator=g) - 0.5
        return (w if weight_offset else w + 1.0).to(BF16)

    return HeadNorm(weight(), weight(), weight_offset, 1e-6)


def run_gqa(
    solution: str | None,
    qkv: torch.Tensor,
    hq: int,
    hkv: int,
    head_dim: int,
    *,
    norm: HeadNorm | None,
    rotary: Rotary | None,
    fmt: KVCacheFormat,
    return_kv: bool,
    slots: torch.Tensor,
    total: int,
    strided_q: bool = False,
) -> list[torch.Tensor]:
    """Query, key cache, value cache, then the returned key and value."""
    q, k, v = split(qkv.clone(), hq, hkv, head_dim)
    if strided_q:
        q = torch.stack(
            (q.view(-1, hq, head_dim), torch.zeros_like(q).view(-1, hq, head_dim)),
            dim=2,
        ).unbind(2)[0]
    k_cache, v_cache = gqa_cache(
        total, hkv, head_dim, qkv.dtype if fmt is KVCacheFormat.NATIVE else FP8
    )
    out = gqa_prologue(
        q,
        k,
        v,
        norm=norm,
        rotary=rotary,
        cache=HeadKVCache(k_cache=k_cache, v_cache=v_cache, scales=None, slots=slots),
        return_kv=return_kv,
        solution=solution,
        override=None,
    )
    assert (out.k is not None) == (out.v is not None) == return_kv
    return [out.q.flatten(1), k_cache, v_cache] + (
        [out.k.flatten(1), out.v.flatten(1)] if return_kv else []
    )


def mla_inputs(tokens: int, heads: int, rank: int, rope: int, seed: int):
    """Absorbed query's non-RoPE part, its RoPE part, and the latent rows."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q_nope = torch.randn(tokens, heads, rank, dtype=BF16, device="cuda", generator=g)
    q_pe = torch.randn(tokens, heads, rope, dtype=BF16, device="cuda", generator=g)
    latent = torch.randn(tokens, rank + rope, dtype=BF16, device="cuda", generator=g)
    return q_nope, q_pe, latent


def mla_query(q_nope: torch.Tensor, rope: int) -> torch.Tensor:
    tokens, heads, rank = q_nope.shape
    query = q_nope.new_empty(tokens, heads, rank + rope)
    query[..., :rank] = q_nope
    return query


def poisoned_latent(total: int, width: int, dtype: torch.dtype) -> torch.Tensor:
    return poisoned((total, 1, width), dtype)


def latent_target(
    kv_cache,
    slots: torch.Tensor,
    sanitize: bool = False,
    write_mask: torch.Tensor | None = None,
) -> LatentKVCache:
    return LatentKVCache(
        kv_cache=kv_cache, sanitize=sanitize, slots=slots, write_mask=write_mask
    )


def _pair_rows(mrope: MRope, half: int) -> list[int]:
    """The T, H or W position row of each rotary pair (Qwen2-VL and Qwen3-VL)."""
    if not mrope.interleaved:
        return [row for row, pairs in enumerate(mrope.section) for _ in range(pairs)]
    _, h, w = mrope.section
    return [
        1 if j % 3 == 1 and j < 3 * h else 2 if j % 3 == 2 and j < 3 * w else 0
        for j in range(half)
    ]


def reference_heads(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    norm: HeadNorm | None,
    rotary: Rotary | None,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64 norm and rotation of ``[num_tokens, heads * head_dim]`` rows, and
    the magnitude of the terms each output sums."""
    x = x.double().reshape(x.shape[0], -1, head_dim)
    if norm is not None:
        rms = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm.eps)
        x = x * rms * (weight.double() + norm.weight_offset)
    magnitude = x.abs()
    if rotary is not None:
        half = rotary.rotary_dim // 2
        table = rotary.cos_sin_cache.double()
        positions = rotary.positions
        if positions.ndim == 2:
            rows = torch.tensor(_pair_rows(rotary.mrope, half) * 2, device="cuda")
            cols = torch.arange(2 * half, device="cuda")
            positions = positions[rows].T
            cos_sin = table[positions, cols]
        else:
            cos_sin = table[positions]
        cos, sin = cos_sin[:, None, :half], cos_sin[:, None, half:]
        if rotary.style is RopeStyle.NEOX:
            i1, i2 = slice(0, half), slice(half, 2 * half)
        else:
            i1, i2 = slice(0, 2 * half, 2), slice(1, 2 * half, 2)
        x1, x2 = x[..., i1].clone(), x[..., i2].clone()
        x[..., i1] = x1 * cos - x2 * sin
        x[..., i2] = x2 * cos + x1 * sin
        magnitude[..., i1] = x1.abs() * cos.abs() + x2.abs() * sin.abs()
        magnitude[..., i2] = x2.abs() * cos.abs() + x1.abs() * sin.abs()
    return x.flatten(1), magnitude.flatten(1)


def assert_rounded_once(
    out: torch.Tensor, ref: torch.Tensor, magnitude: torch.Tensor
) -> None:
    """``out`` is ``ref`` rounded once to its dtype: within half a unit in the
    last place, plus fp32 noise on the ``magnitude`` of the terms it sums."""
    if ref.numel() == 0:
        return
    mantissa_bits, min_exponent = _FLOAT_FORMATS[out.dtype]
    mantissa, exponent = torch.frexp(ref)
    half_ulp = torch.ldexp(
        torch.ones_like(ref), exponent.clamp(min=min_exponent + 1) - mantissa_bits - 2
    )
    # Below a normal power of two the spacing halves, so a result there gets half.
    below = (
        (mantissa.abs() == 0.5)
        & (exponent > min_exponent + 1)
        & (out.double().abs() < ref.abs())
    )
    half_ulp = torch.where(below, half_ulp / 2, half_ulp)
    allowed = torch.where(ref == 0, 0.0, half_ulp) + 2.0**-20 * magnitude
    error = (out.double() - ref).abs()
    worst = int((error - allowed).argmax())
    assert (
        error.flatten()[worst] <= allowed.flatten()[worst]
    ), f"{out.flatten()[worst]} is not {ref.flatten()[worst]} rounded once"


def assert_gqa_rounds_once(
    solution: str,
    head_dim: int,
    rotary_dim: int,
    style: RopeStyle | None,
    weight_offset: float | None,
    mrope: MRope | None,
    fmt: KVCacheFormat,
    dtype: torch.dtype,
) -> None:
    """Query, returned key and value, and both cache rows are the fp64 norm and
    rotation rounded once; 65 tokens span more than one Triton tile."""
    hq, hkv, tokens = 8, 2, 65
    inputs = qkv(tokens, hq, hkv, head_dim, seed=head_dim + rotary_dim, dtype=dtype)
    g = torch.Generator(device="cuda").manual_seed(1)
    norm = None if weight_offset is None else head_norm(head_dim, weight_offset, 2)
    rotary = None
    if style is not None:
        shape = (tokens,) if mrope is None else (3, tokens)
        positions = torch.randint(0, 4096, shape, device="cuda", generator=g)
        rotary = Rotary(cos_sin_cache(rotary_dim), positions, style, mrope)
    q_out, k_cache, v_cache, k_out, v_out = run_gqa(
        solution,
        inputs,
        hq,
        hkv,
        head_dim,
        norm=norm,
        rotary=rotary,
        fmt=fmt,
        return_kv=True,
        slots=torch.arange(tokens, device="cuda"),
        total=tokens,
    )
    q, k, v = split(inputs, hq, hkv, head_dim)
    q_weight = None if norm is None else norm.q_weight
    k_weight = None if norm is None else norm.k_weight
    ref_k = reference_heads(k, k_weight, norm, rotary, head_dim)
    assert_rounded_once(q_out, *reference_heads(q, q_weight, norm, rotary, head_dim))
    assert_rounded_once(k_out, *ref_k)
    assert_rounded_once(k_cache.flatten(1), *ref_k)
    assert_rounded_once(v_out, v.double(), v.double().abs())
    assert_rounded_once(v_cache.flatten(1), v.double(), v.double().abs())
