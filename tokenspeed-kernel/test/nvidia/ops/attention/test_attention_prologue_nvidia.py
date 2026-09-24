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

"""The NVIDIA GQA prologue claims: the Triton kernel matches the CUDA RoPE path
models ran before it byte for byte, the solutions agree, and dispatch is one
Triton launch on every vendor except for MXFP8 caches."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from utils import is_nvidia

if not is_nvidia():
    pytest.skip("NVIDIA GPU required", allow_module_level=True)

import tokenspeed_kernel.ops.attention.prologue as prologue  # noqa: E402
from attention_prologue_reference import (  # noqa: E402
    BF16,
    FP8,
    bytes_equal,
    cos_sin_cache,
    gqa_cache,
    latent_target,
    mla_inputs,
    mla_query,
    poisoned_latent,
    qkv,
    run_gqa,
    slots,
    split,
)
from tokenspeed_kernel.ops.attention.prologue import (  # noqa: E402
    HeadKVCache,
    KVCacheFormat,
    MRope,
    RopeStyle,
    Rotary,
    gqa_prologue,
    mla_prologue,
)
from tokenspeed_kernel.ops.embedding import (  # noqa: E402
    FusedSetKVBufferArg,
    apply_rope,
)
from tokenspeed_kernel.platform import (  # noqa: E402
    ArchVersion,
    PlatformInfo,
    current_platform,
)
from tokenspeed_kernel.selection import select_kernel  # noqa: E402
from tokenspeed_kernel.signature import (  # noqa: E402
    dense_tensor_format,
    format_signature,
)

FORMATS = [KVCacheFormat.NATIVE, KVCacheFormat.FP8]
DTYPES = [torch.bfloat16, torch.float16]


def _rotary(rope: str, head_dim: int, tokens: int) -> Rotary | None:
    g = torch.Generator(device="cuda").manual_seed(3)
    positions = torch.randint(0, 4096, (3, tokens), device="cuda", generator=g)
    half = head_dim // 2
    return {
        "neox": Rotary(cos_sin_cache(head_dim), positions[0], RopeStyle.NEOX, None),
        "gptj": Rotary(cos_sin_cache(head_dim), positions[0], RopeStyle.GPTJ, None),
        "partial": Rotary(cos_sin_cache(half), positions[0], RopeStyle.NEOX, None),
        # The CUDA RoPE's vector loads cannot take this NEOX width; Triton rotates it.
        "narrow": Rotary(cos_sin_cache(48), positions[0], RopeStyle.NEOX, None),
        "nope": None,
        "mrope": Rotary(
            cos_sin_cache(head_dim),
            positions,
            RopeStyle.NEOX,
            MRope((half - 2 * (half // 4), half // 4, half // 4), True),
        ),
    }[rope]


@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("rope", ["neox", "gptj", "partial", "narrow", "nope", "mrope"])
@pytest.mark.parametrize("return_kv", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
def test_unnormed_solutions_agree_byte_for_byte(head_dim, rope, return_kv, dtype):
    """Without a norm the composite (the CUDA RoPE, then the store) rounds once
    too on a native cache, so the two solutions agree to the byte."""
    hq, hkv, tokens, total = 8, 2, 65, 128
    runs = [
        run_gqa(
            solution,
            qkv(tokens, hq, hkv, head_dim, seed=4, dtype=dtype),
            hq,
            hkv,
            head_dim,
            norm=None,
            rotary=_rotary(rope, head_dim, tokens),
            fmt=KVCacheFormat.NATIVE,
            return_kv=return_kv,
            slots=slots(tokens, total, seed=5),
            total=total,
        )
        for solution in ("triton", "composite")
    ]
    for a, b in zip(*runs):
        assert bytes_equal(a, b)


@pytest.mark.parametrize("tokens", [1, 65, 300])
@pytest.mark.parametrize("style", [RopeStyle.NEOX, RopeStyle.GPTJ])
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_is_the_decode_write_models_ran(tokens, style, fmt, dtype):
    """Llama-style decode rotated and stored K/V in one CUDA embedding.rope
    launch; the Triton kernel reproduces its bytes on native and FP8 caches."""
    hq, hkv, dim, total = 8, 2, 128, 512
    inputs = qkv(tokens, hq, hkv, dim, seed=6, dtype=dtype)
    positions = torch.arange(100, 100 + tokens, device="cuda")
    cos_sin = cos_sin_cache(dim)
    loc = slots(tokens, total, seed=7)

    q, k, v = split(inputs.clone(), hq, hkv, dim)
    ref_k, ref_v = gqa_cache(
        total, hkv, dim, dtype if fmt is KVCacheFormat.NATIVE else FP8
    )
    ref_q = torch.empty((tokens, hq * dim), dtype=dtype, device="cuda")
    apply_rope(
        positions,
        q,
        k,
        dim,
        cos_sin,
        is_neox=style is RopeStyle.NEOX,
        fused_set_kv_buffer_arg=FusedSetKVBufferArg(
            value=v.view(tokens, hkv, dim),
            k_buffer=ref_k.view(total, -1),
            v_buffer=ref_v.view(total, -1),
            cache_loc=loc,
        ),
        q_rope_out=ref_q,
    )

    got = run_gqa(
        "triton",
        inputs,
        hq,
        hkv,
        dim,
        norm=None,
        rotary=Rotary(cos_sin, positions, style, None),
        fmt=fmt,
        return_kv=False,
        slots=loc,
        total=total,
    )
    for a, b in zip(got, [ref_q, ref_k, ref_v]):
        assert bytes_equal(a, b)


@pytest.mark.parametrize(
    "tokens,seeds", [(1, range(2)), (9, range(2)), (512, range(6))]
)
@pytest.mark.parametrize(
    "rope,rope_style",
    [(64, RopeStyle.GPTJ), (64, RopeStyle.NEOX), (64, None), (0, None)],
)
@pytest.mark.parametrize("fp8_cache", [True, False])
@pytest.mark.parametrize("sanitize", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
def test_mla_triton_matches_the_composite(
    tokens, seeds, rope, rope_style, fp8_cache, sanitize, dtype
):
    """DeepSeek decode at TP8: 16 heads, a 512-wide latent and 64 RoPE channels."""
    heads, rank, total = 16, 512, 1024
    cache_dtype = FP8 if fp8_cache else dtype
    for seed in seeds:
        inputs = mla_inputs(tokens, heads, rank, rope, seed=15 + seed)
        q_nope, q_pe, latent = (x.to(dtype) for x in inputs)
        if sanitize:
            latent[0, 1] = float("nan")
            latent[-1, 2] = float("inf")
            latent[0, rank:] = torch.finfo(dtype).max
        g = torch.Generator(device="cuda").manual_seed(seed)
        positions = torch.randint(0, 4096, (tokens,), device="cuda", generator=g)
        rotary = (
            None
            if rope_style is None
            else Rotary(cos_sin_cache(rope), positions, rope_style, None)
        )
        loc = slots(tokens, total, seed=seed)

        def run(solution):
            cache = poisoned_latent(total, rank + rope, cache_dtype)
            out = mla_prologue(
                mla_query(q_nope, rope),
                q_pe.clone(),
                latent.clone(),
                expanded=None,
                rotary=rotary,
                cache=latent_target(cache, loc, sanitize),
                solution=solution,
                override=None,
            )
            return out.query, cache

        for a, b in zip(run("triton"), run("composite")):
            assert bytes_equal(a, b)


def _platform(vendor: str) -> PlatformInfo:
    """Another vendor's platform; its own arch keeps its rows out of the NVIDIA
    selection cache entries."""
    if vendor == "nvidia":
        return current_platform()
    return dataclasses.replace(
        current_platform(), vendor=vendor, arch_version=ArchVersion(9, 5)
    )


def _selected(tokens: int, dtype: torch.dtype, platform: PlatformInfo, **change) -> str:
    traits = {
        "head_dim": 128,
        "token_heads": tokens * 32,
        "has_norm": False,
        "kv_format": "native",
        "kv_convert": False,
        "mrope": False,
        "partial_rotary": False,
        "return_kv": False,
        "rope_style": "neox",
    }
    kernel = select_kernel(
        "attention",
        "gqa_prologue",
        format_signature(q=dense_tensor_format(dtype)),
        platform=platform,
        traits={**traits, **change},
    )
    return kernel.name


@pytest.mark.parametrize(
    "tokens,change,expected",
    [
        (1, {}, "triton_gqa_prologue"),
        (4096, {}, "triton_gqa_prologue"),
        (4096, {"has_norm": True}, "triton_gqa_prologue"),
        (4096, {"has_norm": True, "head_dim": 96}, "triton_gqa_prologue"),
        (4096, {"partial_rotary": True}, "triton_gqa_prologue"),
        (4096, {"mrope": True}, "triton_gqa_prologue"),
        (4096, {"rope_style": "none"}, "triton_gqa_prologue"),
        (4096, {"kv_convert": True}, "triton_gqa_prologue"),
        (
            4096,
            {"kv_format": "fp8", "return_kv": True, "rope_style": "gptj"},
            "triton_gqa_prologue",
        ),
        (4, {"kv_format": "mxfp8"}, "composite_gqa_prologue"),
        (4096, {"has_norm": True, "kv_format": "mxfp8"}, "composite_gqa_prologue"),
    ],
)
@pytest.mark.parametrize("vendor", ["nvidia", "amd"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_gqa_dispatch_is_one_triton_launch_except_mxfp8(
    tokens, change, expected, vendor, dtype
):
    """The token-tiled Triton kernel beats the CUDA RoPE path at every size, so
    only MXFP8 caches (its store kernels) reach the composite."""
    assert _selected(tokens, dtype, _platform(vendor), **change) == expected


def test_cuda_rope_flushes_the_subnormals_the_triton_kernel_keeps():
    """Besides NaN payloads, the CUDA RoPE (the composite) and the Triton kernel
    differ only by what a flush can move: less than 2^-125, or one ulp through a
    broken rounding tie."""
    hq, hkv, dim, tokens = 8, 2, 128, 65
    inputs = qkv(tokens, hq, hkv, dim, seed=17)
    inputs[:, ::3] *= 1e-39
    inputs[:, 1::3] *= 1e-37
    # Token 1 plants an fp32 tie, then a bf16 tie, that the flushed x2 * sin breaks.
    table = cos_sin_cache(dim)
    table[1, : dim // 2] = 105691 * 2.0**-17
    table[1, dim // 2 :] = (1 - table[1, 0] ** 2).sqrt()
    pairs = inputs[1, : (hq + hkv) * dim].view(hq + hkv, 2, dim // 2)
    pairs[:, 0], pairs[:, 1] = 173 * 2.0**-7, 2.0**-126
    args = dict(
        norm=None,
        rotary=Rotary(table, torch.arange(tokens, device="cuda"), RopeStyle.NEOX, None),
        fmt=KVCacheFormat.NATIVE,
        return_kv=True,
        slots=torch.arange(tokens, device="cuda"),
        total=tokens,
    )
    triton = run_gqa("triton", inputs, hq, hkv, dim, **args)
    cuda = run_gqa("composite", inputs, hq, hkv, dim, **args)
    flushed = ties = 0
    for a, b in zip(triton, cuda):
        differ = a.view(torch.int16) != b.view(torch.int16)
        a, b = a[differ].float(), b[differ].float()
        exponent = torch.frexp(torch.maximum(a.abs(), b.abs()))[1]
        ulp = torch.ldexp(torch.ones_like(a), exponent - 8)
        assert ((a - b).abs() <= 2**-125 + ulp).all()
        flushed += int(differ.numel())
        ties += int((a.abs() > 1).sum())
    assert flushed and ties


def test_an_override_this_platform_cannot_run_raises(monkeypatch):
    """The Triton kernel has no Ascend build; naming it there raises."""
    ascend = _platform("ascend")
    monkeypatch.setattr(prologue, "current_platform", lambda: ascend)
    prologue._select.cache_clear()
    hq, hkv, dim, tokens = 4, 2, 128, 130
    q, k, v = split(qkv(tokens, hq, hkv, dim, seed=15), hq, hkv, dim)
    k_cache, v_cache = gqa_cache(256, hkv, dim, BF16)
    try:
        with pytest.raises(ValueError, match="does not serve"):
            gqa_prologue(
                q,
                k,
                v,
                norm=None,
                rotary=Rotary(
                    cos_sin_cache(dim),
                    torch.arange(tokens, device="cuda"),
                    RopeStyle.NEOX,
                    None,
                ),
                cache=HeadKVCache(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    scales=None,
                    slots=slots(tokens, 256, seed=16),
                ),
                return_kv=False,
                override="triton_gqa_prologue",
                solution=None,
            )
    finally:
        prologue._select.cache_clear()
