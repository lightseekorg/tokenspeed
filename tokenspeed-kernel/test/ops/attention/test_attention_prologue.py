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

"""The portable attention prologue: validation, dispatch, the composite's step
sequence, and the single rounding of the Triton GQA and MLA solutions."""

from __future__ import annotations

import dataclasses
import itertools

import pytest
import torch
from attention_prologue_reference import (
    BF16,
    FP8,
    POISON,
    assert_gqa_rounds_once,
    assert_rounded_once,
    bytes_equal,
    cos_sin_cache,
    gqa_cache,
    head_norm,
    latent_target,
    mla_inputs,
    mla_query,
    poisoned_latent,
    qkv,
    reference_heads,
    run_gqa,
    slots,
    split,
)
from tokenspeed_kernel.ops.attention.prologue import (
    HeadKVCache,
    HeadNorm,
    KVCacheFormat,
    MLAExpandedKV,
    MRope,
    MXFP8Scales,
    PerTokenHeadPlanes,
    RopeStyle,
    Rotary,
    gqa_prologue,
    mla_prologue,
    qk_norm_rope,
)
from tokenspeed_kernel.ops.embedding import apply_rope, apply_rope_mla
from tokenspeed_kernel.ops.kvcache.triton import (
    fused_fp8_set_kv_buffer,
    quantize_store_kv_mxfp8,
    set_mla_kv_buffer_triton,
    store_kv_cache,
)
from tokenspeed_kernel.ops.layernorm import qk_rmsnorm
from tokenspeed_kernel.ops.quantization.triton import fp8_quantize
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


@pytest.mark.parametrize("tokens", [1, 7, 64])
@pytest.mark.parametrize(
    "head_dim,rotary_dim", [(128, 128), (128, 64), (96, 64), (64, 64)]
)
@pytest.mark.parametrize("weight_offset", [None, 0.0, 1.0])
@pytest.mark.parametrize("fmt", [KVCacheFormat.NATIVE, KVCacheFormat.FP8])
@pytest.mark.parametrize("hkv", [2, 3])
def test_gqa_composite_runs_the_step_kernels(
    tokens, head_dim, rotary_dim, weight_offset, fmt, hkv
):
    hq, total = 4 * hkv, 256
    cache_dtype = BF16 if fmt is KVCacheFormat.NATIVE else FP8
    inputs = qkv(tokens, hq, hkv, head_dim, seed=1)
    positions = torch.arange(3, 3 + tokens, device="cuda")
    cos_sin = cos_sin_cache(rotary_dim)
    norm = None if weight_offset is None else head_norm(head_dim, weight_offset, 2)
    rows = slots(tokens, total, seed=2)

    q, k, v = split(inputs.clone(), hq, hkv, head_dim)
    if norm is not None:
        q, k = qk_rmsnorm(
            q,
            k,
            norm.q_weight,
            norm.k_weight,
            norm.eps,
            weight_offset=weight_offset,
        )
    q, k = apply_rope(positions, q, k, head_dim, cos_sin, is_neox=True)
    ref_k, ref_v = gqa_cache(total, hkv, head_dim, cache_dtype)
    k3 = k.view(tokens, hkv, head_dim)
    v3 = v.view(tokens, hkv, head_dim)
    if fmt is KVCacheFormat.NATIVE:
        store_kv_cache(k3, v3, ref_k, ref_v, rows)
    else:
        fused_fp8_set_kv_buffer(k3, v3, ref_k, ref_v, rows, page_size=1)
        assert bytes_equal(ref_k[rows], k3.to(FP8))

    got = run_gqa(
        "composite",
        inputs,
        hq,
        hkv,
        head_dim,
        norm=norm,
        rotary=Rotary(cos_sin, positions, RopeStyle.NEOX, None),
        fmt=fmt,
        return_kv=True,
        slots=rows,
        total=total,
    )
    for a, b in zip(got, [q, ref_k, ref_v, k, v]):
        assert bytes_equal(a, b)


@pytest.mark.parametrize("solution", [None, "composite"])
def test_the_gemma_offset_is_formed_in_fp32(solution):
    """``1 + w`` of bf16 weights is never rounded to bf16 before it scales the head."""
    tokens, hq, hkv, dim = 37, 8, 2, 128
    q, k, v = split(qkv(tokens, hq, hkv, dim, seed=52), hq, hkv, dim)
    norm = head_norm(dim, 1.0, seed=53)
    assert norm.q_weight.dtype == BF16
    k_cache, v_cache = gqa_cache(64, hkv, dim, BF16)
    out = gqa_prologue(
        q.clone(),
        k.clone(),
        v,
        norm=norm,
        rotary=None,
        cache=HeadKVCache(k_cache, v_cache, None, slots(tokens, 64, seed=54)),
        return_kv=True,
        solution=solution,
        override=None,
    )
    assert_rounded_once(out.q, *reference_heads(q, norm.q_weight, norm, None, dim))
    assert_rounded_once(out.k, *reference_heads(k, norm.k_weight, norm, None, dim))


def _triton_round_once_cases():
    styles = [RopeStyle.NEOX, RopeStyle.GPTJ]
    formats = [KVCacheFormat.NATIVE, KVCacheFormat.FP8]
    for (dim, rotary_dim), style, offset, fmt in itertools.product(
        [(128, 128), (128, 64), (256, 64), (64, 64)], styles, [None, 0.0, 1.0], formats
    ):
        yield dim, rotary_dim, style, offset, None, fmt
    for offset, fmt in itertools.product([None, 0.0], formats):
        yield 128, 128, None, offset, None, fmt
    for fmt in formats:
        yield 96, 96, RopeStyle.NEOX, 1.0, None, fmt
    for (dim, rotary_dim, section), interleaved, offset, fmt in itertools.product(
        [(128, 128, (24, 20, 20)), (256, 64, (11, 11, 10))],
        [True, False],
        [None, 0.0],
        formats,
    ):
        yield dim, rotary_dim, RopeStyle.NEOX, offset, MRope(section, interleaved), fmt


@pytest.mark.parametrize(
    "head_dim,rotary_dim,style,weight_offset,mrope,fmt",
    list(_triton_round_once_cases()),
)
@pytest.mark.parametrize("dtype", [BF16, torch.float16])
def test_triton_rounds_once(
    head_dim, rotary_dim, style, weight_offset, mrope, fmt, dtype
):
    assert_gqa_rounds_once(
        "triton", head_dim, rotary_dim, style, weight_offset, mrope, fmt, dtype
    )


@pytest.mark.parametrize("solution", [None, "composite"])
def test_a_zero_offset_keeps_negative_zero_weights(solution):
    """A -0.0 weight flips each normalized value's sign, as the multiplication does."""
    tokens, hq, hkv, dim = 4, 2, 1, 64
    q, k, v = split(qkv(tokens, hq, hkv, dim, seed=60), hq, hkv, dim)
    weight = torch.full((dim,), -0.0, dtype=BF16, device="cuda")
    k_cache, v_cache = gqa_cache(8, hkv, dim, BF16)
    out = gqa_prologue(
        q.clone(),
        k.clone(),
        v,
        norm=HeadNorm(weight, weight, 0.0, 1e-6),
        rotary=None,
        cache=HeadKVCache(k_cache, v_cache, None, slots(tokens, 8, seed=61)),
        return_kv=True,
        solution=solution,
        override=None,
    )
    for x, y in ((q, out.q), (k, out.k)):
        assert torch.equal(y, torch.zeros_like(y))
        assert torch.equal(torch.signbit(y), ~torch.signbit(x))


@pytest.mark.parametrize("solution", [None, "composite"])
@pytest.mark.parametrize("interleaved", [True, False])
def test_mrope_zero_sections_are_plain_rope_by_the_t_row(solution, interleaved):
    tokens, hq, hkv, dim = 6, 4, 2, 64
    inputs = qkv(tokens, hq, hkv, dim, seed=90)
    g = torch.Generator(device="cuda").manual_seed(91)
    rows = torch.randint(0, 4096, (3, tokens), device="cuda", generator=g)
    mrope = MRope((dim // 2, 0, 0), interleaved)
    runs = [
        run_gqa(
            solution,
            inputs,
            hq,
            hkv,
            dim,
            norm=None,
            rotary=Rotary(cos_sin_cache(dim), positions, RopeStyle.NEOX, section),
            fmt=KVCacheFormat.NATIVE,
            return_kv=True,
            slots=slots(tokens, 8, seed=92),
            total=8,
        )
        for positions, section in ((rows, mrope), (rows[0].contiguous(), None))
    ]
    for a, b in zip(*runs):
        assert bytes_equal(a, b)


@pytest.mark.parametrize("solution", [None, "composite"])
@pytest.mark.parametrize("scale", [0.0, 1e-4])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_the_norm_epsilon_is_applied(solution, scale, eps):
    """Heads whose mean square is near or below eps, which randn inputs never reach."""
    tokens, hq, hkv, dim = 6, 4, 2, 128
    q, k, v = split(qkv(tokens, hq, hkv, dim, seed=130) * scale, hq, hkv, dim)
    norm = dataclasses.replace(head_norm(dim, 0.0, seed=131), eps=eps)
    k_cache, v_cache = gqa_cache(16, hkv, dim, BF16)
    out = gqa_prologue(
        q.clone(),
        k.clone(),
        v,
        norm=norm,
        rotary=None,
        cache=HeadKVCache(k_cache, v_cache, None, slots(tokens, 16, seed=132)),
        return_kv=True,
        solution=solution,
        override=None,
    )
    assert torch.isfinite(out.q).all() and torch.isfinite(out.k).all()
    assert_rounded_once(out.q, *reference_heads(q, norm.q_weight, norm, None, dim))
    assert_rounded_once(out.k, *reference_heads(k, norm.k_weight, norm, None, dim))


def test_a_q_pe_view_with_singleton_strides_is_the_rope_channels():
    """Strides of size-1 dimensions place no element, so they need not match."""
    tokens, heads, rank, rope = 4, 1, 512, 64
    q_nope, q_pe, latent = mla_inputs(tokens, heads, rank, rope, seed=195)
    query = mla_query(q_nope, rope)
    query[..., rank:] = q_pe
    cache = poisoned_latent(8, rank + rope, BF16)
    out = mla_prologue(
        query,
        query[..., rank:].squeeze(1).unsqueeze(1),
        latent,
        expanded=None,
        rotary=None,
        cache=latent_target(cache, torch.arange(tokens, device="cuda")),
        solution=None,
        override=None,
    )
    assert bytes_equal(out.query[..., rank:], q_pe)


def test_an_mla_override_past_its_token_head_limit_raises():
    tokens, heads, rank, rope = 2049, 16, 512, 64
    q_nope, q_pe, latent = mla_inputs(tokens, heads, rank, rope, seed=145)
    with pytest.raises(ValueError, match="does not serve"):
        mla_prologue(
            mla_query(q_nope, rope),
            q_pe,
            latent,
            expanded=None,
            rotary=None,
            cache=latent_target(
                poisoned_latent(tokens, rank + rope, BF16),
                torch.arange(tokens, device="cuda"),
            ),
            solution=None,
            override="triton_mla_prologue",
        )


@pytest.mark.parametrize("override", ["composite_mla_prologue", "triton_mla_prologue"])
def test_an_override_must_serve_the_request(override):
    """Another mode's kernel raises."""
    tokens = 4
    q, k, v = split(qkv(tokens, 4, 2, 64, seed=140), 4, 2, 64)
    k_cache, v_cache = gqa_cache(256, 2, 64, BF16)
    with pytest.raises(ValueError, match="does not serve"):
        gqa_prologue(
            q,
            k,
            v,
            norm=None,
            rotary=Rotary(
                cos_sin_cache(64),
                torch.arange(tokens, device="cuda"),
                RopeStyle.NEOX,
                None,
            ),
            cache=HeadKVCache(k_cache, v_cache, None, slots(tokens, 256, seed=141)),
            return_kv=False,
            solution=None,
            override=override,
        )


@pytest.mark.parametrize(
    "head_dim,rotary_dim,style",
    [(128, 128, RopeStyle.NEOX), (128, 128, RopeStyle.GPTJ), (128, 64, RopeStyle.NEOX)],
)
@pytest.mark.parametrize("dtype", [BF16, torch.float16])
def test_triton_and_the_composite_agree_unnormed_on_a_native_cache(
    head_dim, rotary_dim, style, dtype
):
    """Both round once there, so a row's bytes never depend on which one ran;
    ~2M rotated pairs, so a different fma association flips some bytes; one
    row goes unwritten."""
    hq, hkv, tokens, total = 14, 2, 2048, 2048
    g = torch.Generator(device="cuda").manual_seed(4)
    rotary = Rotary(
        cos_sin_cache(rotary_dim),
        torch.randint(0, 4096, (tokens,), device="cuda", generator=g),
        style,
        None,
    )
    runs = [
        run_gqa(
            solution,
            qkv(tokens, hq, hkv, head_dim, seed=5, dtype=dtype),
            hq,
            hkv,
            head_dim,
            norm=None,
            rotary=rotary,
            fmt=KVCacheFormat.NATIVE,
            return_kv=True,
            slots=slots(tokens - 1, total, seed=6),
            total=total,
        )
        for solution in ("triton", "composite")
    ]
    for a, b in zip(*runs):
        assert bytes_equal(a, b)


def test_gqa_composite_writes_mxfp8():
    """Inkling: normed, NoPE, 128-wide heads."""
    tokens, written, hq, hkv, dim, page_tokens = 7, 5, 4, 2, 128, 128
    rows = 16 * page_tokens
    q, k, v = split(qkv(tokens, hq, hkv, dim, seed=3), hq, hkv, dim)
    norm = head_norm(dim, 0.0, seed=4)
    loc = slots(written, rows, seed=5)

    def planes():
        data = torch.zeros(rows, hkv, dim, dtype=FP8, device="cuda")
        scale = torch.zeros(
            rows // page_tokens,
            hkv,
            1,
            32,
            4,
            4,
            dtype=torch.float8_e8m0fnu,
            device="cuda",
        )
        return data, scale

    ref_q, ref_k = qk_rmsnorm(
        q.clone(), k.clone(), norm.q_weight, norm.k_weight, 1e-6, weight_offset=0.0
    )
    (ref_kc, ref_ks), (ref_vc, ref_vs) = planes(), planes()
    quantize_store_kv_mxfp8(
        ref_k[:written].view(written, hkv, dim),
        v[:written].view(written, hkv, dim),
        ref_kc,
        ref_vc,
        ref_ks,
        ref_vs,
        loc,
        page_tokens=page_tokens,
    )

    (kc, ks), (vc, vs) = planes(), planes()
    out = gqa_prologue(
        q,
        k,
        v,
        norm=norm,
        rotary=None,
        cache=HeadKVCache(
            k_cache=kc,
            v_cache=vc,
            scales=MXFP8Scales(ks, vs, page_tokens),
            slots=loc,
        ),
        return_kv=False,
        solution=None,
        override=None,
    )
    assert bytes_equal(out.q, ref_q)
    for a, b in zip((kc, ks, vc, vs), (ref_kc, ref_ks, ref_vc, ref_vs)):
        assert bytes_equal(a, b)


@pytest.mark.parametrize(
    "head_dim,rotary_dim,section", [(128, 128, (24, 20, 20)), (256, 64, (11, 11, 10))]
)
@pytest.mark.parametrize("interleaved", [True, False])
@pytest.mark.parametrize("rope_style", [RopeStyle.NEOX, RopeStyle.GPTJ])
def test_mrope_rotates_each_pair_by_its_section_row(
    head_dim, rotary_dim, section, interleaved, rope_style
):
    """Image tokens carry distinct T/H/W rows; Qwen3.5 hands its query as a
    strided view of the interleaved q|gate rows."""
    tokens, hq, hkv = 23, 8, 2
    inputs = qkv(tokens, hq, hkv, head_dim, seed=6)
    g = torch.Generator(device="cuda").manual_seed(7)
    rotary = Rotary(
        cos_sin_cache(rotary_dim),
        torch.randint(0, 4096, (3, tokens), device="cuda", generator=g),
        rope_style,
        MRope(section, interleaved),
    )
    got_q, got_k, _ = run_gqa(
        "composite",
        inputs,
        hq,
        hkv,
        head_dim,
        norm=None,
        rotary=rotary,
        fmt=KVCacheFormat.NATIVE,
        return_kv=False,
        slots=torch.arange(tokens, device="cuda"),
        total=tokens,
        strided_q=True,
    )
    q, k, _ = split(inputs, hq, hkv, head_dim)
    assert_rounded_once(got_q, *reference_heads(q, None, None, rotary, head_dim))
    assert_rounded_once(
        got_k.flatten(1), *reference_heads(k, None, None, rotary, head_dim)
    )


@pytest.mark.parametrize("weight_offset", [None, 0.0])
@pytest.mark.parametrize("fmt", [KVCacheFormat.NATIVE, KVCacheFormat.FP8])
def test_mrope_text_rows_are_plain_rope(weight_offset, fmt):
    tokens, hq, hkv, dim, total = 20, 4, 2, 128, 32
    inputs = qkv(tokens, hq, hkv, dim, seed=8)
    positions = torch.arange(5, 5 + tokens, device="cuda")
    cos_sin = cos_sin_cache(dim)
    args = dict(
        norm=None if weight_offset is None else head_norm(dim, weight_offset, 9),
        fmt=fmt,
        return_kv=True,
        slots=slots(tokens, total, seed=10),
        total=total,
    )
    plain = run_gqa(
        None,
        inputs,
        hq,
        hkv,
        dim,
        rotary=Rotary(cos_sin, positions, RopeStyle.NEOX, None),
        **args,
    )
    for rows in (positions, positions.expand(3, -1).contiguous()):
        rotary = Rotary(cos_sin, rows, RopeStyle.NEOX, MRope((24, 20, 20), True))
        for a, b in zip(
            run_gqa(None, inputs, hq, hkv, dim, rotary=rotary, **args), plain
        ):
            assert bytes_equal(a, b)


@pytest.mark.parametrize("solution", [None, "composite"])
@pytest.mark.parametrize("cache_dtype", [BF16, FP8])
def test_gqa_writes_only_the_slotted_rows(solution, cache_dtype):
    tokens, rows, hq, hkv, dim, total = 12, 5, 4, 2, 128, 64
    q, k, v = split(qkv(tokens, hq, hkv, dim, seed=11), hq, hkv, dim)
    loc = slots(rows, total, seed=12)
    k_cache, v_cache = gqa_cache(total, hkv, dim, cache_dtype)
    before_k, before_v = k_cache.clone(), v_cache.clone()
    out = gqa_prologue(
        q,
        k,
        v,
        norm=None,
        rotary=None,
        cache=HeadKVCache(
            k_cache=k_cache,
            v_cache=v_cache,
            scales=None,
            slots=loc,
        ),
        return_kv=False,
        solution=solution,
        override=None,
    )
    untouched = torch.ones(total, dtype=torch.bool, device="cuda")
    untouched[loc] = False
    assert out.k is None and out.v is None
    assert bytes_equal(k_cache[untouched], before_k[untouched])
    assert bytes_equal(v_cache[untouched], before_v[untouched])
    assert bytes_equal(k_cache[loc], k[:rows].view(rows, hkv, dim).to(cache_dtype))
    assert bytes_equal(v_cache[loc], v[:rows].view(rows, hkv, dim).to(cache_dtype))


def test_int32_indices_give_the_int64_bytes():
    """Slots and positions arrive as int32 or int64; every solution reads both alike."""
    tokens, hq, hkv, dim, total = 6, 4, 2, 64, 16
    inputs = qkv(tokens, hq, hkv, dim, seed=150)
    positions = torch.arange(3, 3 + tokens, device="cuda")
    loc = slots(tokens, total, seed=151)
    runs = [
        run_gqa(
            None,
            inputs,
            hq,
            hkv,
            dim,
            norm=None,
            rotary=Rotary(
                cos_sin_cache(dim), positions.to(dtype), RopeStyle.NEOX, None
            ),
            fmt=KVCacheFormat.NATIVE,
            return_kv=True,
            slots=loc.to(dtype),
            total=total,
        )
        for dtype in (torch.int64, torch.int32)
    ]
    for a, b in zip(*runs):
        assert bytes_equal(a, b)
    heads, rank, rope = 4, 512, 64
    q_nope, q_pe, latent = mla_inputs(tokens, heads, rank, rope, seed=152)
    outs = []
    for dtype in (torch.int64, torch.int32):
        cache = poisoned_latent(total, rank + rope, BF16)
        out = mla_prologue(
            mla_query(q_nope, rope),
            q_pe.clone(),
            latent.clone(),
            expanded=None,
            rotary=Rotary(
                cos_sin_cache(rope), positions.to(dtype), RopeStyle.GPTJ, None
            ),
            cache=latent_target(cache, loc.to(dtype)),
            solution=None,
            override=None,
        )
        outs.append((out.query, cache))
    for a, b in zip(*outs):
        assert bytes_equal(a, b)


def test_triton_reads_query_heads_past_int32_offsets():
    """A head index times its stride can pass 2^31 elements; the kernel widens both."""
    hq, hkv, dim, stride = 17, 1, 64, 2**27
    base = torch.empty((hq - 1) * stride + dim, dtype=BF16, device="cuda")
    q = base.as_strided((1, hq, dim), (hq * stride, stride, 1))
    g = torch.Generator(device="cuda").manual_seed(153)
    q.copy_(torch.randn(1, hq, dim, dtype=BF16, device="cuda", generator=g))
    _, k, v = split(qkv(1, hq, hkv, dim, seed=154), hq, hkv, dim)
    k_cache, v_cache = gqa_cache(4, hkv, dim, BF16)
    out = gqa_prologue(
        q,
        k,
        v,
        norm=None,
        rotary=None,
        cache=HeadKVCache(k_cache, v_cache, None, torch.arange(1, device="cuda")),
        return_kv=False,
        solution="triton",
        override=None,
    )
    assert bytes_equal(out.q, q.reshape(1, -1))


@pytest.mark.parametrize(
    "solution,fmt,strided",
    [
        ("triton", BF16, "q_pe"),
        ("composite", BF16, "q_pe"),
        ("composite", FP8, "q_pe"),
        ("composite", FP8, "query"),
    ],
)
def test_mla_solutions_read_query_heads_past_int32_offsets(solution, fmt, strided):
    """A head index times its stride passes 2^31 at head 16; a sentinel sits where
    the wrapped offset would land, in q_pe or in the query the FP8 writer reads."""
    heads, rank, rope, stride, guard = 17, 512, 64, 2**27, 2**31
    width = rope if strided == "q_pe" else rank + rope
    base = torch.zeros(guard + (heads - 1) * stride + width, dtype=BF16, device="cuda")
    base[:width] = -7.0
    view = base[guard:].as_strided((1, heads, width), (heads * stride, stride, 1))
    view.copy_(torch.arange(1, heads + 1, device="cuda").to(BF16).view(1, heads, 1))
    if strided == "q_pe":
        query, q_pe = (
            mla_query(torch.zeros(1, heads, rank, dtype=BF16, device="cuda"), rope),
            view,
        )
    else:
        query, q_pe = view, view[..., rank:]
    latent = torch.ones(1, rank + rope, dtype=BF16, device="cuda")
    cache = poisoned_latent(4, rank + rope, fmt)
    out = mla_prologue(
        query,
        q_pe,
        latent,
        expanded=None,
        rotary=None,
        cache=latent_target(cache, torch.arange(1, device="cuda")),
        solution=solution,
        override=None,
    )
    assert bytes_equal(out.query[..., rank:], q_pe.to(out.query.dtype))
    if strided == "query":
        assert bytes_equal(out.query[..., :rank], view[..., :rank].to(out.query.dtype))


def test_triton_mla_reads_int32_positions_past_two_to_the_25():
    """An int32 position times the table row stride passes 2^31 at 2^25 rows."""
    tokens, heads, rank, rope, position = 1, 4, 512, 64, 2**25
    q_nope, q_pe, latent = mla_inputs(tokens, heads, rank, rope, seed=160)
    table = torch.zeros(position + 1, rope, dtype=torch.float32, device="cuda")
    table[position] = cos_sin_cache(rope)[1]
    outs = []
    for dtype in (torch.int64, torch.int32):
        cache = poisoned_latent(4, rank + rope, BF16)
        out = mla_prologue(
            mla_query(q_nope, rope),
            q_pe.clone(),
            latent.clone(),
            expanded=None,
            rotary=Rotary(
                table,
                torch.tensor([position], device="cuda", dtype=dtype),
                RopeStyle.NEOX,
                None,
            ),
            cache=latent_target(cache, torch.arange(tokens, device="cuda")),
            solution="triton",
            override=None,
        )
        outs.append((out.query, cache))
    for a, b in zip(*outs):
        assert bytes_equal(a, b)
    assert not bytes_equal(outs[0][0][..., rank:], q_pe)


def test_triton_reads_mrope_rows_past_int32_offsets():
    """A position row's stride times its index passes 2^31 for the W row."""
    tokens, hq, hkv, dim, stride = 2, 4, 2, 64, 2**30
    inputs = qkv(tokens, hq, hkv, dim, seed=161)
    rows = torch.tensor([[5, 6], [7, 8], [9, 10]], device="cuda", dtype=torch.int32)
    base = torch.zeros(2 * stride + tokens, dtype=torch.int32, device="cuda")
    wide = base.as_strided((3, tokens), (stride, 1))
    wide.copy_(rows)
    runs = [
        run_gqa(
            "triton",
            inputs,
            hq,
            hkv,
            dim,
            norm=None,
            rotary=Rotary(
                cos_sin_cache(dim),
                positions,
                RopeStyle.NEOX,
                MRope((0, 0, dim // 2), False),
            ),
            fmt=KVCacheFormat.NATIVE,
            return_kv=True,
            slots=torch.arange(tokens, device="cuda"),
            total=tokens,
        )
        for positions in (rows, wide)
    ]
    for a, b in zip(*runs):
        assert bytes_equal(a, b)


@pytest.mark.parametrize("solution", [None, "triton", "composite"])
@pytest.mark.parametrize("tokens", [4, 65])
def test_fp16_rows_round_once_into_a_bf16_cache(solution, tokens):
    """An fp16 model keeps its activations; the fused kernels round the bf16 cache
    row once from fp32, the composite casts its fp16 result as the pool did.
    65 tokens: more than one Triton tile."""
    hq, hkv, dim = 8, 2, 64
    q, k, v = split(
        qkv(tokens, hq, hkv, dim, seed=170, dtype=torch.float16), hq, hkv, dim
    )
    rotary = Rotary(
        cos_sin_cache(dim), torch.arange(tokens, device="cuda"), RopeStyle.NEOX, None
    )
    k_cache, v_cache = gqa_cache(128, hkv, dim, BF16)
    loc = slots(tokens, 128, seed=171)
    out = gqa_prologue(
        q.clone(),
        k.clone(),
        v,
        norm=None,
        rotary=rotary,
        cache=HeadKVCache(k_cache, v_cache, None, loc),
        return_kv=True,
        solution=solution,
        override=None,
    )
    assert out.q.dtype is out.k.dtype is torch.float16
    ref_k = reference_heads(k, None, None, rotary, dim)
    assert_rounded_once(out.k, *ref_k)
    if solution == "composite":
        assert bytes_equal(k_cache[loc].flatten(1), out.k.to(BF16))
    else:
        assert_rounded_once(k_cache[loc].flatten(1), *ref_k)
    assert bytes_equal(v_cache[loc].flatten(1), v.to(BF16))


@pytest.mark.parametrize("solution", ["triton", "composite"])
def test_fp16_mla_rows_round_once_into_a_bf16_cache(solution):
    """The Triton write rounds the rotated fp16 RoPE part once into bf16; the
    composite casts its fp16 rotation, as the pool did."""
    tokens, heads, rank, rope = 3, 4, 512, 64
    g = torch.Generator(device="cuda").manual_seed(172)
    q_nope = torch.randn(
        tokens, heads, rank, dtype=torch.float16, device="cuda", generator=g
    )
    q_pe = torch.randn(
        tokens, heads, rope, dtype=torch.float16, device="cuda", generator=g
    )
    latent = torch.randn(
        tokens, rank + rope, dtype=torch.float16, device="cuda", generator=g
    )
    rotary = Rotary(
        cos_sin_cache(rope),
        torch.arange(5, 5 + tokens, device="cuda"),
        RopeStyle.NEOX,
        None,
    )
    cache = poisoned_latent(8, rank + rope, BF16)
    loc = slots(tokens, 8, seed=173)
    out = mla_prologue(
        mla_query(q_nope, rope),
        q_pe.clone(),
        latent.clone(),
        expanded=None,
        rotary=rotary,
        cache=latent_target(cache, loc),
        solution=solution,
        override=None,
    )
    assert out.query.dtype is torch.float16
    assert bytes_equal(cache[loc, 0, :rank], latent[:, :rank].to(BF16))
    if solution == "triton":
        ref = reference_heads(latent[:, rank:], None, None, rotary, rope)
        assert_rounded_once(cache[loc, 0, rank:], *ref)
        assert_rounded_once(
            out.query[..., rank:].flatten(1),
            *reference_heads(q_pe.flatten(1), None, None, rotary, rope),
        )


def test_a_single_kv_head_cache_may_carry_any_head_stride():
    """TP-sharded Llama has one KV head; a size-1 head dimension is packed at any stride."""
    tokens, hq, hkv, dim, total = 4, 4, 1, 64, 8
    q, k, v = split(qkv(tokens, hq, hkv, dim, seed=180), hq, hkv, dim)
    loc = slots(tokens, total, seed=181)
    caches = []
    for stride in (dim, 0):
        k_cache, v_cache = (
            torch.zeros(total, dim, dtype=BF16, device="cuda").as_strided(
                (total, 1, dim), (dim, stride, 1)
            )
            for _ in range(2)
        )
        gqa_prologue(
            q,
            k,
            v,
            norm=None,
            rotary=None,
            cache=HeadKVCache(k_cache, v_cache, None, loc),
            return_kv=False,
            solution=None,
            override=None,
        )
        caches.append((k_cache, v_cache))
    for a, b in zip(*caches):
        assert bytes_equal(a, b)


def test_composite_reads_kv_rows_past_int32_offsets():
    """The native store's row index times the source row stride passes 2^31 at row
    16; a sentinel sits where the wrapped offset would land."""
    tokens, hq, hkv, dim, stride, guard = 17, 1, 1, 128, 2**27, 2**31
    base = torch.zeros(guard + (tokens - 1) * stride + dim, dtype=BF16, device="cuda")
    base[:dim] = -7.0
    k = base[guard:].as_strided((tokens, dim), (stride, 1))
    k.copy_(torch.arange(1, tokens + 1, device="cuda").to(BF16).view(tokens, 1))
    q, _, v = split(qkv(tokens, hq, hkv, dim, seed=196), hq, hkv, dim)
    k_cache, v_cache = gqa_cache(32, hkv, dim, BF16)
    loc = slots(tokens, 32, seed=197)
    gqa_prologue(
        q,
        k,
        v,
        norm=None,
        rotary=None,
        cache=HeadKVCache(k_cache, v_cache, None, loc),
        return_kv=False,
        solution="composite",
        override=None,
    )
    assert bytes_equal(k_cache[loc].flatten(1), k)


def test_composite_mla_reads_latent_rows_past_int32_offsets():
    """The latent scatter's row index times the row stride passes 2^31 at row 16;
    a sentinel sits where the wrapped offset would land."""
    tokens, rank, rope, stride, guard = 17, 512, 64, 2**27, 2**31
    width = rank + rope
    base = torch.zeros(guard + (tokens - 1) * stride + width, dtype=BF16, device="cuda")
    base[:width] = -7.0
    latent = base[guard:].as_strided((tokens, width), (stride, 1))
    latent.copy_(torch.arange(1, tokens + 1, device="cuda").to(BF16).view(tokens, 1))
    q_nope = torch.zeros(tokens, 1, rank, dtype=BF16, device="cuda")
    query = mla_query(q_nope, rope)
    cache = poisoned_latent(32, width, BF16)
    loc = slots(tokens, 32, seed=190)
    mla_prologue(
        query,
        query[..., rank:],
        latent,
        expanded=None,
        rotary=None,
        cache=latent_target(cache, loc),
        solution="composite",
        override=None,
    )
    assert bytes_equal(cache[loc, 0], latent)


@pytest.mark.parametrize("fmt", [KVCacheFormat.NATIVE, KVCacheFormat.MXFP8])
def test_the_composite_writes_a_prefill_past_65535_tokens(fmt):
    """Tokens sit on the launch grid's first axis, whose limit is 2^31, not 65535:
    one launch over 65536 rows writes what two launches over the halves write."""
    tokens, hq, hkv, dim = 65536, 1, 1, 128
    q, k, v = split(qkv(tokens, hq, hkv, dim, seed=191), hq, hkv, dim)

    def write(row_ranges):
        k_cache, v_cache = gqa_cache(
            tokens, hkv, dim, BF16 if fmt is KVCacheFormat.NATIVE else FP8
        )
        planes = [
            torch.zeros(
                tokens // 128,
                hkv,
                1,
                32,
                4,
                4,
                dtype=torch.float8_e8m0fnu,
                device="cuda",
            )
            for _ in range(2)
        ]
        scales = (
            None
            if fmt is KVCacheFormat.NATIVE
            else MXFP8Scales(planes[0], planes[1], 128)
        )
        for a, b in row_ranges:
            gqa_prologue(
                q[a:b],
                k[a:b],
                v[a:b],
                norm=None,
                rotary=None,
                cache=HeadKVCache(
                    k_cache, v_cache, scales, torch.arange(a, b, device="cuda")
                ),
                return_kv=False,
                solution="composite",
                override=None,
            )
        return k_cache, v_cache, *planes

    for a, b in zip(
        write([(0, tokens)]), write([(0, tokens // 2), (tokens // 2, tokens)])
    ):
        assert bytes_equal(a, b)


@pytest.mark.parametrize("layout", ["transposed", "padded", "tight"])
def test_non_interleaved_strided_queries_are_accepted(layout):
    """Heads outside tokens, rows wider than their content, or padded heads whose
    next row starts right after the last head's channels: no shared address."""
    tokens, hq, hkv, dim = 2, 3, 1, 64
    if layout == "transposed":
        q = torch.randn(hq, tokens, dim, dtype=BF16, device="cuda").transpose(0, 1)
    elif layout == "padded":
        q = torch.randn(tokens, hq + 1, dim, dtype=BF16, device="cuda")[:, :hq]
    else:
        row = (hq - 1) * (dim + 8) + dim
        base = torch.randn(tokens * row, dtype=BF16, device="cuda")
        q = base.as_strided((tokens, hq, dim), (row, dim + 8, 1))
    _, k, v = split(qkv(tokens, hq, hkv, dim, seed=156), hq, hkv, dim)
    outs = [
        gqa_prologue(
            x,
            k,
            v,
            norm=None,
            rotary=None,
            cache=HeadKVCache(
                *gqa_cache(4, hkv, dim, BF16), None, slots(tokens, 4, seed=157)
            ),
            return_kv=False,
            solution=None,
            override=None,
        ).q
        for x in (q, q.contiguous())
    ]
    assert bytes_equal(*outs)


@pytest.mark.parametrize("fp8_cache", [True, False])
def test_expanded_values_from_a_wider_head_buffer(fp8_cache):
    """Expanded keys and values may be head slices of a wider buffer."""
    tokens, heads, wide, nope, rope, rank, v_dim, total = 2, 3, 8, 128, 64, 512, 128, 16
    g = torch.Generator(device="cuda").manual_seed(158)
    q = torch.randn(tokens, heads, nope + rope, dtype=BF16, device="cuda", generator=g)
    latent = torch.randn(tokens, rank + rope, dtype=BF16, device="cuda", generator=g)
    k_wide = torch.randn(tokens, wide, nope, dtype=BF16, device="cuda", generator=g)
    v_wide = torch.randn(tokens, wide, v_dim, dtype=BF16, device="cuda", generator=g)
    loc = slots(tokens, total, seed=159)

    def run(k_nope, value):
        cache = poisoned_latent(total, rank + rope, FP8 if fp8_cache else BF16)
        out = mla_prologue(
            q.clone(),
            q[..., nope:].clone(),
            latent.clone(),
            expanded=MLAExpandedKV(k_nope, value),
            rotary=None,
            cache=latent_target(cache, loc),
            solution=None,
            override=None,
        )
        return out.query, out.key, out.value, cache

    for a, b in zip(
        run(k_wide[:, :heads], v_wide[:, :heads]),
        run(k_wide[:, :heads].contiguous(), v_wide[:, :heads].contiguous()),
    ):
        assert bytes_equal(a, b)


@pytest.mark.parametrize("tokens,rows", [(1, 1), (9, 4), (9, 0)])
def test_triton_writes_only_the_slotted_rows_of_a_strided_query(tokens, rows):
    """Qwen3.5 decode: a strided query, multimodal rows and padded tokens."""
    hq, hkv, dim, total = 4, 2, 256, 16
    inputs = qkv(tokens, hq, hkv, dim, seed=8)
    norm = head_norm(dim, 0.0, seed=9)
    g = torch.Generator(device="cuda").manual_seed(10)
    rotary = Rotary(
        cos_sin_cache(64),
        torch.randint(0, 4096, (3, tokens), device="cuda", generator=g),
        RopeStyle.NEOX,
        MRope((11, 11, 10), True),
    )
    loc = slots(rows, total, seed=11)
    q_out, k_cache, v_cache = run_gqa(
        "triton",
        inputs,
        hq,
        hkv,
        dim,
        norm=norm,
        rotary=rotary,
        fmt=KVCacheFormat.NATIVE,
        return_kv=False,
        slots=loc,
        total=total,
        strided_q=True,
    )

    q, k, v = split(inputs, hq, hkv, dim)
    assert_rounded_once(q_out, *reference_heads(q, norm.q_weight, norm, rotary, dim))
    ref_k, magnitude = reference_heads(k, norm.k_weight, norm, rotary, dim)
    assert_rounded_once(k_cache[loc].flatten(1), ref_k[:rows], magnitude[:rows])
    assert torch.equal(v_cache[loc].flatten(1), v[:rows])
    untouched = torch.ones(total, dtype=torch.bool, device="cuda")
    untouched[loc] = False
    for cache in (k_cache, v_cache):
        assert (cache[untouched].view(torch.uint8) == POISON).all()


def test_qk_norm_rope_is_the_prologue_without_a_cache_write():
    """MiniMax-M3's indexer normalizes and rotates keys it caches itself."""
    tokens, hq, hk, dim = 9, 4, 1, 128
    inputs = qkv(tokens, hq, hk, dim, seed=41)
    norm = head_norm(dim, 1.0, seed=42)
    rotary = Rotary(
        cos_sin_cache(dim), torch.arange(tokens, device="cuda"), RopeStyle.NEOX, None
    )
    q, k, _ = split(inputs.clone(), hq, hk, dim)
    got_q, got_k = qk_norm_rope(q, k, head_dim=dim, norm=norm, rotary=rotary)
    ref = run_gqa(
        None,
        inputs,
        hq,
        hk,
        dim,
        norm=norm,
        rotary=rotary,
        fmt=KVCacheFormat.NATIVE,
        return_kv=True,
        slots=torch.arange(tokens, device="cuda"),
        total=tokens,
    )
    assert bytes_equal(got_q, ref[0]) and bytes_equal(got_k, ref[3])


@pytest.mark.parametrize("tokens", [1, 9, 64])
@pytest.mark.parametrize("rope_style", [RopeStyle.GPTJ, RopeStyle.NEOX, None])
@pytest.mark.parametrize("fp8_cache", [True, False])
def test_mla_composite_runs_the_deepseek_decode_steps(tokens, rope_style, fp8_cache):
    heads, rank, rope, total = 16, 512, 64, 256
    cache_dtype = FP8 if fp8_cache else BF16
    q_nope, q_pe, latent = mla_inputs(tokens, heads, rank, rope, seed=13)
    positions = torch.arange(50, 50 + tokens, device="cuda")
    cos_sin = cos_sin_cache(rope) if rope_style is not None else None
    loc = slots(tokens, total, seed=14)
    rotary = (
        None if rope_style is None else Rotary(cos_sin, positions, rope_style, None)
    )

    ref_cache = poisoned_latent(total, rank + rope, cache_dtype)
    ref_query = mla_query(q_nope, rope)
    key = latent.clone().unsqueeze(1)
    if fp8_cache:
        ref_query, ref_key = apply_rope_mla(
            positions=positions,
            q_rope=q_pe.clone(),
            k_rope=key[..., rank:],
            q_nope=ref_query[..., :rank],
            k_nope=key[..., :rank],
            cos_sin_cache=cos_sin,
            is_neox=rope_style is RopeStyle.NEOX,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
        )
        set_mla_kv_buffer_triton(
            ref_cache, loc, ref_key[..., :rank], ref_key[..., rank:]
        )
    else:
        if rope_style is not None:
            qp, _ = apply_rope(
                positions,
                q_pe.clone(),
                key[..., rank:],
                rope,
                cos_sin,
                is_neox=rope_style is RopeStyle.NEOX,
            )
            ref_query[..., rank:].copy_(qp)
        else:
            ref_query[..., rank:] = q_pe
        set_mla_kv_buffer_triton(ref_cache, loc, key[..., :rank], key[..., rank:])

    new_cache = poisoned_latent(total, rank + rope, cache_dtype)
    out = mla_prologue(
        mla_query(q_nope, rope),
        q_pe.clone(),
        latent.clone(),
        expanded=None,
        rotary=rotary,
        cache=latent_target(new_cache, loc),
        solution="composite",
        override=None,
    )
    assert bytes_equal(out.query, ref_query)
    assert bytes_equal(new_cache, ref_cache)


@pytest.mark.parametrize("rope_style", [RopeStyle.GPTJ, RopeStyle.NEOX])
@pytest.mark.parametrize("cache_dtype", [BF16, FP8])
def test_mla_triton_rounds_once(rope_style, cache_dtype):
    """The one-launch write rotates in fp32 and rounds each query and cache row once."""
    tokens, heads, rank, rope, total = 64, 16, 512, 64, 128
    q_nope, q_pe, latent = mla_inputs(tokens, heads, rank, rope, seed=43)
    rotary = Rotary(
        cos_sin_cache(rope), torch.arange(tokens, device="cuda"), rope_style, None
    )
    loc = slots(tokens, total, seed=44)
    cache = poisoned_latent(total, rank + rope, cache_dtype)
    out = mla_prologue(
        mla_query(q_nope, rope),
        q_pe.clone(),
        latent.clone(),
        expanded=None,
        rotary=rotary,
        cache=latent_target(cache, loc),
        solution="triton",
        override=None,
    )
    q_pe_ref, q_pe_mag = reference_heads(q_pe.flatten(1), None, None, rotary, rope)
    k_pe_ref, k_pe_mag = reference_heads(latent[:, rank:], None, None, rotary, rope)
    nope = q_nope.double()
    query_ref = torch.cat((nope, q_pe_ref.view(tokens, heads, rope)), -1)
    query_mag = torch.cat((nope.abs(), q_pe_mag.view(tokens, heads, rope)), -1)
    row_ref = torch.cat((latent[:, :rank].double(), k_pe_ref), -1)
    row_mag = torch.cat((latent[:, :rank].double().abs(), k_pe_mag), -1)
    assert_rounded_once(out.query, query_ref, query_mag)
    assert_rounded_once(cache[loc, 0], row_ref, row_mag)


@pytest.mark.parametrize("case", ["expanded", "per_token_head", "partial_write"])
def test_mla_triton_declines_what_it_does_not_cover(case):
    tokens, heads, rank, rope = 4, 4, 512, 64
    q_nope, q_pe, latent = mla_inputs(tokens, heads, rank, rope, seed=21)
    loc = slots(tokens - (case == "partial_write"), 16, seed=22)
    cache = latent_target(poisoned_latent(16, rank + rope, BF16), loc)
    if case == "per_token_head":
        planes = PerTokenHeadPlanes(
            latent=torch.zeros(16, 1, rank, dtype=torch.uint8, device="cuda"),
            scale=torch.zeros(16, 1, 1, dtype=torch.float32, device="cuda"),
            rope=torch.zeros(16, 1, rope, dtype=BF16, device="cuda"),
        )
        cache = latent_target(planes, loc)
    expanded = None
    query = mla_query(q_nope, rope)
    if case == "expanded":
        query = query[..., rank - 128 :].contiguous()
        expanded = MLAExpandedKV(
            k_nope=torch.zeros(tokens, heads, 128, dtype=BF16, device="cuda"),
            value=torch.zeros(tokens, heads, 128, dtype=BF16, device="cuda"),
        )
    kwargs = dict(expanded=expanded, rotary=None, cache=cache)
    with pytest.raises(NoKernelFoundError):
        mla_prologue(
            query,
            q_pe,
            latent,
            **kwargs,
            solution="triton",
            override=None,
        )
    with pytest.raises(ValueError, match="does not serve"):
        mla_prologue(
            query,
            q_pe,
            latent,
            **kwargs,
            override="triton_mla_prologue",
            solution=None,
        )


@pytest.mark.parametrize(
    "token_heads,expected",
    [
        (32768, "triton_mla_prologue"),
        (32769, "composite_mla_prologue"),
    ],
)
def test_mla_selection_follows_the_measured_limit(token_heads, expected):
    """DeepSeek at TP8 with 16 heads reaches 32768 at 2048 tokens."""
    traits = {
        "token_heads": token_heads,
        "expanded": False,
        "full_write": True,
        "kv_format": "fp8",
        "rope_style": "gptj",
        "sanitize": False,
    }
    kernel = select_kernel(
        "attention",
        "mla_prologue",
        format_signature(query=dense_tensor_format(BF16)),
        traits=traits,
    )
    assert kernel.name == expected


@pytest.mark.parametrize("style", [RopeStyle.NEOX, RopeStyle.GPTJ])
@pytest.mark.parametrize("dtype", [BF16, torch.float16])
def test_mla_triton_and_the_composite_agree_on_a_native_cache(style, dtype):
    """~1M rotated pairs at the token-head limit, so an association change shows;
    q_pe aliases the query's RoPE channels, as DeepSeek's decode passes it."""
    tokens, heads, rank, rope, total = 2048, 16, 512, 64, 2048
    q_nope, q_pe, latent = (
        x.to(dtype) for x in mla_inputs(tokens, heads, rank, rope, seed=97)
    )
    g = torch.Generator(device="cuda").manual_seed(98)
    positions = torch.randint(0, 4096, (tokens,), device="cuda", generator=g)
    loc = torch.randperm(total, device="cuda", generator=g)

    def run(solution):
        cache = poisoned_latent(total, rank + rope, dtype)
        query = mla_query(q_nope, rope)
        query[..., rank:] = q_pe
        out = mla_prologue(
            query,
            query[..., rank:],
            latent.clone(),
            expanded=None,
            rotary=Rotary(cos_sin_cache(rope), positions, style, None),
            cache=latent_target(cache, loc),
            solution=solution,
            override=None,
        )
        return out.query, cache

    for a, b in zip(run("triton"), run("composite")):
        assert bytes_equal(a, b)


def test_mla_zero_width_rope_stores_a_plain_cast():
    """GLM-5.3-Flash keeps no RoPE channels; its FP8 cache holds the cast latent."""
    tokens, heads, rank, total = 5, 4, 512, 16
    q_nope, q_pe, latent = mla_inputs(tokens, heads, rank, 0, seed=23)
    loc = slots(tokens, total, seed=24)
    cache = poisoned_latent(total, rank, FP8)
    mla_prologue(
        mla_query(q_nope, 0),
        q_pe,
        latent,
        expanded=None,
        rotary=None,
        cache=latent_target(cache, loc, sanitize=True),
        solution="composite",
        override=None,
    )
    assert bytes_equal(cache[loc, 0], latent.to(FP8))


def test_mla_per_token_head_scales_each_latent_row_into_fp8():
    tokens, rows, heads, rank, rope, total = 6, 4, 4, 512, 64, 32
    q_nope, q_pe, latent = mla_inputs(tokens, heads, rank, rope, seed=25)
    latent[1, 3] = float("nan")
    loc = slots(rows, total, seed=26)

    def planes():
        return PerTokenHeadPlanes(
            latent=torch.zeros(total, 1, rank, dtype=torch.uint8, device="cuda"),
            scale=torch.zeros(total, 1, 1, dtype=torch.float32, device="cuda"),
            rope=torch.zeros(total, 1, rope, dtype=BF16, device="cuda"),
        )

    ref = planes()
    k_nope = torch.nan_to_num(latent[:rows, :rank].unsqueeze(1))
    k_rope = torch.nan_to_num(latent[:rows, rank:].unsqueeze(1))
    k_lora = k_nope.float()
    scale = k_lora.abs().amax(dim=-1, keepdim=True).clamp(1e-26) / 448.0
    ref.latent[loc] = (k_lora / scale).to(FP8).view(torch.uint8)
    ref.scale[loc] = scale
    ref.rope[loc] = (k_rope.float() / scale).to(BF16)

    got = planes()
    mla_prologue(
        mla_query(q_nope, rope),
        q_pe,
        latent,
        expanded=None,
        rotary=None,
        cache=latent_target(got, loc, sanitize=True),
        solution=None,
        override=None,
    )
    for name in ("latent", "scale", "rope"):
        assert bytes_equal(getattr(got, name), getattr(ref, name))


def test_mla_writes_only_the_committed_rows():
    tokens, rows, heads, rank, rope, total = 10, 4, 8, 512, 64, 64
    q_nope, q_pe, latent = mla_inputs(tokens, heads, rank, rope, seed=27)
    loc = slots(rows, total, seed=28)
    cache = poisoned_latent(total, rank + rope, BF16)
    before = cache.clone()
    mla_prologue(
        mla_query(q_nope, rope),
        q_pe,
        latent,
        expanded=None,
        rotary=None,
        cache=latent_target(cache, loc),
        solution=None,
        override=None,
    )
    untouched = torch.ones(total, dtype=torch.bool, device="cuda")
    untouched[loc] = False
    assert torch.equal(cache[untouched], before[untouched])
    assert torch.equal(cache[loc, 0], latent[:rows])


@pytest.mark.parametrize("tokens", [1, 37])
@pytest.mark.parametrize("rope_style", [RopeStyle.GPTJ, RopeStyle.NEOX, None])
@pytest.mark.parametrize("fp8_cache", [True, False])
@pytest.mark.parametrize("heads", [1, 16])
def test_mla_expanded_runs_the_deepseek_prefill_steps(
    tokens, rope_style, fp8_cache, heads
):
    """Non-absorbed prefill attends per-head keys up-projected from the latent."""
    nope, rope, rank, v_dim, total = 128, 64, 512, 128, 128
    cache_dtype = FP8 if fp8_cache else BF16
    g = torch.Generator(device="cuda").manual_seed(29)
    q = torch.randn(tokens, heads, nope + rope, dtype=BF16, device="cuda", generator=g)
    latent = torch.randn(tokens, rank + rope, dtype=BF16, device="cuda", generator=g)
    kv = torch.randn(
        tokens, heads, nope + v_dim, dtype=BF16, device="cuda", generator=g
    )
    k_nope, v = kv.split([nope, v_dim], dim=-1)
    positions = torch.arange(20, 20 + tokens, device="cuda")
    cos_sin = cos_sin_cache(rope) if rope_style is not None else None
    loc = slots(tokens, total, seed=30)
    rotary = (
        None if rope_style is None else Rotary(cos_sin, positions, rope_style, None)
    )

    ref_q, ref_latent = q.clone(), latent.clone()
    kv_a, k_pe = ref_latent.split([rank, rope], dim=-1)
    k_pe = k_pe.unsqueeze(1)
    q_nope, q_pe = ref_q.split([nope, rope], dim=-1)
    ref_cache = poisoned_latent(total, rank + rope, cache_dtype)
    if fp8_cache:
        ref_q, ref_k = apply_rope_mla(
            positions=positions,
            q_rope=q_pe,
            k_rope=k_pe if rope_style is None else k_pe.expand(-1, heads, -1),
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin,
            is_neox=rope_style is RopeStyle.NEOX,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
        )
        ref_v = fp8_quantize(v)
        set_mla_kv_buffer_triton(
            ref_cache, loc, kv_a.unsqueeze(1), ref_k[:, 0:1, nope:]
        )
    else:
        if rope_style is not None:
            q_pe, k_pe = apply_rope(
                positions,
                q_pe,
                k_pe,
                rope,
                cos_sin,
                is_neox=rope_style is RopeStyle.NEOX,
            )
        ref_q[..., nope:] = q_pe
        ref_k = torch.empty_like(ref_q)
        ref_k[..., :nope] = k_nope
        ref_k[..., nope:] = k_pe
        ref_v = v
        set_mla_kv_buffer_triton(ref_cache, loc, kv_a.unsqueeze(1), k_pe)

    new_q = q.clone()
    new_cache = poisoned_latent(total, rank + rope, cache_dtype)
    out = mla_prologue(
        new_q,
        new_q[..., nope:],
        latent.clone(),
        expanded=MLAExpandedKV(k_nope=k_nope, value=v),
        rotary=rotary,
        cache=latent_target(new_cache, loc),
        solution=None,
        override=None,
    )
    assert bytes_equal(out.query, ref_q)
    assert bytes_equal(out.key, ref_k)
    assert bytes_equal(out.value, ref_v)
    assert bytes_equal(new_cache, ref_cache)


@pytest.mark.parametrize("fp8_cache", [True, False])
def test_mla_absorbed_returns_no_key(fp8_cache):
    q_nope, q_pe, latent = mla_inputs(3, 4, 512, 64, seed=31)
    out = mla_prologue(
        mla_query(q_nope, 64),
        q_pe,
        latent,
        expanded=None,
        rotary=None,
        cache=latent_target(
            poisoned_latent(8, 576, FP8 if fp8_cache else BF16), slots(3, 8, 32)
        ),
        solution=None,
        override=None,
    )
    assert out.key is None and out.value is None


@pytest.mark.parametrize("mrope", [False, True])
def test_prologues_accept_zero_tokens(mrope):
    cos_sin = cos_sin_cache(64)
    empty = torch.empty(0, dtype=torch.int64, device="cuda")
    positions = empty.expand(3, 0).contiguous() if mrope else empty
    q, k, v = split(qkv(0, 4, 2, 64, seed=33), 4, 2, 64)
    k_cache, v_cache = gqa_cache(8, 2, 64, BF16)
    gqa = gqa_prologue(
        q,
        k,
        v,
        norm=head_norm(64, 0.0, seed=34),
        rotary=Rotary(
            cos_sin,
            positions,
            RopeStyle.NEOX,
            MRope((12, 10, 10), True) if mrope else None,
        ),
        cache=HeadKVCache(
            k_cache=k_cache,
            v_cache=v_cache,
            scales=None,
            slots=empty,
        ),
        return_kv=True,
        solution=None,
        override=None,
    )
    assert gqa.q.shape == (0, 4 * 64) and gqa.k.shape == (0, 2 * 64)
    q_nope, q_pe, latent = mla_inputs(0, 4, 512, 64, seed=35)
    mla = mla_prologue(
        mla_query(q_nope, 64),
        q_pe,
        latent,
        expanded=None,
        rotary=Rotary(cos_sin, empty, RopeStyle.GPTJ, None),
        cache=latent_target(poisoned_latent(8, 576, BF16), empty),
        solution=None,
        override=None,
    )
    assert mla.query.shape == (0, 4, 576)


def _gqa_request(**change) -> dict:
    tokens, hq, hkv, dim = 4, 4, 2, 64
    q, k, v = split(qkv(tokens, hq, hkv, dim, seed=36), hq, hkv, dim)
    k_cache, v_cache = gqa_cache(8, hkv, dim, BF16)
    request = dict(
        q=q,
        k=k,
        v=v,
        norm=head_norm(dim, 0.0, seed=37),
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
            slots=slots(tokens, 8, seed=38),
        ),
        return_kv=False,
        solution=None,
        override=None,
    )
    return _changed(request, change)


def _changed(request: dict, change: dict) -> dict:
    """Apply ``field=f(request)`` and ``field__attr=f(field)`` edits in order."""
    for name, value in change.items():
        field, _, attr = name.partition("__")
        request[field] = (
            dataclasses.replace(request[field], **{attr: value(request[field])})
            if attr
            else value(request)
        )
    return request


def _strided(x: torch.Tensor) -> torch.Tensor:
    """The same values with a last-dimension stride of 2."""
    return x.repeat_interleave(2, -1)[..., ::2]


def _mrope(rotary: Rotary, section: tuple[int, ...]) -> Rotary:
    rows = rotary.positions.expand(3, -1).contiguous()
    return dataclasses.replace(rotary, positions=rows, mrope=MRope(section, False))


@pytest.mark.parametrize(
    "change,message",
    [
        # fp16 or bf16 activations, even through an override.
        (dict(q=lambda r: r["q"].float()), "fp16 or bf16"),
        (dict(q=lambda r: r["q"].double()), "fp16 or bf16"),
        (dict(q=lambda r: r["q"].to(FP8)), "fp16 or bf16"),
        (dict(q=lambda r: r["q"].to(torch.int16)), "fp16 or bf16"),
        (
            dict(
                q=lambda r: r["q"].float(),
                override=lambda r: "triton_gqa_prologue",
            ),
            "fp16 or bf16",
        ),
        # Query rank.
        (dict(q=lambda r: r["q"].view(4, 4, 1, 64)), "is not \\[tokens"),
        (dict(q=lambda r: r["q"][0, 0]), "is not \\[tokens"),
        # Caches: dense rows of packed heads, one geometry and dtype.
        (dict(cache__k_cache=lambda c: c.k_cache.unsqueeze(1)), "slots, heads"),
        (dict(cache__k_cache=lambda c: c.k_cache.flatten(1)), "slots, heads"),
        (dict(cache__v_cache=lambda c: c.v_cache[..., :32]), "one geometry"),
        (dict(cache__v_cache=lambda c: c.v_cache[:, :1].contiguous()), "one geometry"),
        (dict(cache__v_cache=lambda c: c.v_cache[:2]), "one geometry"),
        (dict(cache__v_cache=lambda c: torch.cat([c.v_cache] * 2)), "one geometry"),
        (dict(cache__v_cache=lambda c: c.v_cache.view(16, 1, 64)), "one geometry"),
        (dict(cache__v_cache=lambda c: c.v_cache.to(FP8)), "one geometry"),
        (
            dict(cache__k_cache=lambda c: torch.cat([c.k_cache] * 2, -1)[..., :64]),
            "one geometry",
        ),
        (dict(cache__k_cache=lambda c: _strided(c.k_cache)), "one geometry"),
        (dict(cache__v_cache=lambda c: _strided(c.v_cache)), "one geometry"),
        (dict(cache__k_cache=lambda c: _strided(c.k_cache[:, :1])), "one geometry"),
        (
            dict(
                cache__k_cache=lambda c: _strided(c.k_cache[:, :1]),
                cache__v_cache=lambda c: _strided(c.v_cache[:, :1]),
            ),
            "one geometry",
        ),
        (
            dict(cache__v_cache=lambda c: torch.cat([c.v_cache] * 2, -1)[..., :64]),
            "one geometry",
        ),
        # Query: dense head_dim-wide heads in 2-D or 3-D rows.
        (dict(q=lambda r: _strided(r["q"])), "wide heads"),
        (dict(q=lambda r: r["q"][:, :200]), "wide heads"),
        (dict(q=lambda r: r["q"].view(4, 2, 128)), "wide heads"),
        # Keys and values: dense 2-D rows of the cache heads.
        (dict(k=lambda r: r["k"].unsqueeze(-1)), "cache heads"),
        (dict(k=lambda r: r["k"][0, 0]), "cache heads"),
        (dict(k=lambda r: r["k"][:, :64]), "cache heads"),
        (dict(v=lambda r: r["v"][:, :64]), "cache heads"),
        (dict(k=lambda r: _strided(r["k"])), "cache heads"),
        (dict(v=lambda r: _strided(r["v"])), "cache heads"),
        # One row count and one dtype.
        (dict(k=lambda r: r["k"][:3]), "same number of rows"),
        (dict(v=lambda r: r["v"][:3]), "same number of rows"),
        (dict(k=lambda r: r["k"][:3], v=lambda r: r["v"][:3]), "same number of rows"),
        (dict(k=lambda r: torch.cat([r["k"], r["k"][:1]])), "same number of rows"),
        (
            dict(
                k=lambda r: torch.cat([r["k"], r["k"][:1]]),
                v=lambda r: torch.cat([r["v"], r["v"][:1]]),
            ),
            "same number of rows",
        ),
        (dict(k=lambda r: r["k"].half()), "share a dtype"),
        (dict(v=lambda r: r["v"].half()), "share a dtype"),
        (dict(k=lambda r: r["k"].half(), v=lambda r: r["v"].half()), "share a dtype"),
        # Nothing a solution may write holds two elements at one address.
        (dict(k=lambda r: r["k"][:1].expand(4, -1)), "share addresses"),
        (dict(v=lambda r: r["v"][:1].expand(4, -1)), "share addresses"),
        (dict(q=lambda r: r["q"][:1].expand(4, -1)), "share addresses"),
        # Rows on 16-byte boundaries, which the CUDA kernels read in vectors.
        (dict(k=lambda r: r["k"].new_zeros(4, 129)[:, 1:]), "16-byte"),
        (dict(q=lambda r: r["q"].new_zeros(4, 4, 65)[..., 1:]), "16-byte"),
        (
            dict(cache__k_cache=lambda c: c.k_cache[:1].expand(8, -1, -1)),
            "share addresses",
        ),
        (
            dict(cache__v_cache=lambda c: c.v_cache[:1].expand(8, -1, -1)),
            "share addresses",
        ),
        (
            dict(
                cache__k_cache=lambda c: c.k_cache.flatten().as_strided(
                    (8, 2, 64), (64, 64, 1)
                )
            ),
            "share addresses",
        ),
        # Norm weights: dense [head_dim].
        (dict(norm__q_weight=lambda n: n.q_weight[:32]), "norm weights"),
        (dict(norm__q_weight=lambda n: n.q_weight.view(1, -1)), "norm weights"),
        (dict(norm__k_weight=lambda n: _strided(n.k_weight)), "norm weights"),
        # Positions, the cos/sin table and M-RoPE sections.
        (dict(rotary__positions=lambda r: r.positions[:3]), "dense rows"),
        (
            dict(rotary__positions=lambda r: torch.cat([r.positions, r.positions[:1]])),
            "dense rows",
        ),
        (dict(rotary__positions=lambda r: r.positions.repeat(2)[::2]), "dense rows"),
        (dict(rotary__positions=lambda r: r.positions.view(1, 1, -1)), "dense rows"),
        (dict(rotary__positions=lambda r: r.positions.float()), "dense rows"),
        (dict(rotary__cos_sin_cache=lambda r: r.cos_sin_cache.to(BF16)), "fp32 rows"),
        (dict(rotary__cos_sin_cache=lambda r: r.cos_sin_cache.double()), "fp32 rows"),
        (dict(rotary__cos_sin_cache=lambda r: r.cos_sin_cache[None]), "fp32 rows"),
        (dict(rotary__cos_sin_cache=lambda r: r.cos_sin_cache[0]), "fp32 rows"),
        (
            dict(rotary__cos_sin_cache=lambda r: r.cos_sin_cache.T.contiguous().T),
            "fp32 rows",
        ),
        (
            dict(rotary__cos_sin_cache=lambda r: r.cos_sin_cache[:, :63].contiguous()),
            "odd",
        ),
        (dict(rotary__positions=lambda r: r.positions.expand(3, -1)), "T/H/W"),
        (
            dict(
                rotary=lambda r: dataclasses.replace(
                    _mrope(r["rotary"], (16, 8, 8)),
                    positions=r["rotary"].positions.expand(2, -1).contiguous(),
                )
            ),
            "T/H/W",
        ),
        (dict(rotary=lambda r: _mrope(r["rotary"], (16, 16))), "sections"),
        (dict(rotary=lambda r: _mrope(r["rotary"], (8, 8, 8))), "sections"),
        (dict(rotary=lambda r: _mrope(r["rotary"], (40, -4, -4))), "sections"),
        (dict(rotary=lambda r: _mrope(r["rotary"], (20, 8, 8))), "sections"),
        (dict(rotary=lambda r: _mrope(r["rotary"], (8, 8, 8, 8))), "sections"),
        (
            dict(
                rotary=lambda r: dataclasses.replace(
                    _mrope(r["rotary"], (16, 8, 8)),
                    positions=r["rotary"].positions.expand(4, -1).contiguous(),
                )
            ),
            "T/H/W",
        ),
        (dict(rotary__cos_sin_cache=lambda r: cos_sin_cache(128)), "exceeds"),
        # Write slots: a dense 1-D vector of at most one slot per token.
        (dict(cache__slots=lambda c: _strided(c.slots)), "dense vector"),
        (
            dict(cache__slots=lambda c: torch.arange(4, device="cuda").view(2, 2).T),
            "dense vector",
        ),
        (dict(cache__slots=lambda c: torch.cat([c.slots, c.slots[:1]])), "cache slots"),
        (dict(cache__slots=lambda c: c.slots.float()), "dense vector"),
        # A native cache holds the activation dtype.
        (
            dict(
                cache__k_cache=lambda c: c.k_cache.half(),
                cache__v_cache=lambda c: c.v_cache.half(),
            ),
            "native cache",
        ),
    ],
)
def test_gqa_entry_rejects_malformed_requests(change, message):
    with pytest.raises(ValueError, match=message):
        gqa_prologue(**_gqa_request(**change))


def _planes(latent_width: int, rope_width: int) -> PerTokenHeadPlanes:
    return PerTokenHeadPlanes(
        latent=torch.zeros(8, 1, latent_width, dtype=torch.uint8, device="cuda"),
        scale=torch.zeros(8, 1, 1, device="cuda"),
        rope=torch.zeros(8, 1, rope_width, dtype=BF16, device="cuda"),
    )


def _mxfp8_request(
    dim=128, cache_dtype=FP8, rows=256, page_tokens=128, k_plane=None, v_plane=None
) -> dict:
    tokens, hq, hkv = 4, 4, 2
    q, k, v = split(qkv(tokens, hq, hkv, dim, seed=3), hq, hkv, dim)
    k_cache, v_cache = gqa_cache(rows, hkv, dim, cache_dtype)
    planes = [
        torch.zeros(
            -(-rows // 128), hkv, 1, 32, 4, 4, dtype=torch.float8_e8m0fnu, device="cuda"
        )
        for _ in range(2)
    ]
    scales = MXFP8Scales(
        planes[0] if k_plane is None else k_plane(planes[0]),
        planes[1] if v_plane is None else v_plane(planes[1]),
        page_tokens,
    )
    return dict(
        q=q,
        k=k,
        v=v,
        norm=None,
        rotary=None,
        cache=HeadKVCache(k_cache, v_cache, scales, slots(tokens, rows, seed=4)),
        return_kv=False,
        solution=None,
        override=None,
    )


@pytest.mark.parametrize(
    "change,message",
    [
        (dict(cache_dtype=BF16), "128-wide FP8"),
        (dict(cache_dtype=torch.float8_e5m2), "128-wide FP8"),
        (dict(dim=64), "128-wide FP8"),
        (dict(dim=256), "128-wide FP8"),
        (dict(page_tokens=64), "positive multiple of 128"),
        (dict(page_tokens=-128), "positive multiple of 128"),
        (dict(page_tokens=0), "positive multiple of 128"),
        (dict(k_plane=lambda p: p.view(torch.uint8)), "scale planes"),
        (dict(v_plane=lambda p: torch.zeros(p.numel(), device="cuda")), "scale planes"),
        (dict(k_plane=lambda p: torch.stack([p, p], -1)[..., 0]), "scale planes"),
        (dict(k_plane=lambda p: p[:1].clone()), "scale planes"),
        (dict(v_plane=lambda p: p[:1].clone()), "scale planes"),
        (dict(k_plane=lambda p: p.flatten()[:-1].clone()), "scale planes"),
        (dict(v_plane=lambda p: p.flatten()[:-4].clone()), "scale planes"),
        (
            dict(k_plane=lambda p: torch.cat([p.new_zeros(1), p.flatten()])[1:]),
            "scale planes",
        ),
        (
            dict(v_plane=lambda p: torch.cat([p.flatten(), p.new_zeros(1)])),
            "scale planes",
        ),
        # 200 rows span two pages; planes sized for the rows alone are short.
        (
            dict(
                rows=200,
                k_plane=lambda p: p.flatten()[: 200 * 2 * 128 // 32].clone(),
                v_plane=lambda p: p.flatten()[: 200 * 2 * 128 // 32].clone(),
            ),
            "scale planes",
        ),
    ],
)
def test_gqa_entry_rejects_malformed_mxfp8_caches(change, message):
    with pytest.raises(ValueError, match=message):
        gqa_prologue(**_mxfp8_request(**change))


def test_the_mxfp8_request_builder_is_valid():
    gqa_prologue(**_mxfp8_request())


def _mla_request(**change) -> dict:
    tokens, heads, rank, rope = 3, 4, 512, 64
    q_nope, q_pe, latent = mla_inputs(tokens, heads, rank, rope, seed=39)
    request = dict(
        query=mla_query(q_nope, rope),
        q_pe=q_pe,
        latent_cache=latent,
        expanded=None,
        rotary=Rotary(
            cos_sin_cache(rope),
            torch.arange(tokens, device="cuda"),
            RopeStyle.GPTJ,
            None,
        ),
        cache=latent_target(
            poisoned_latent(8, rank + rope, BF16), slots(tokens, 8, seed=40)
        ),
        solution=None,
        override=None,
    )
    return _changed(request, change)


def _expanded(k_nope, value) -> dict:
    """Non-absorbed attention with keys and values built from the query's first 128 channels."""
    return dict(
        expanded=lambda r: MLAExpandedKV(
            k_nope(r["query"][..., :128]), value(r["query"][..., :128])
        ),
        query=lambda r: r["query"][..., 384:],
    )


def _plane_cache(latent_width: int, rope_width: int):
    return lambda r: latent_target(
        _planes(latent_width, rope_width), r["cache"].slots, sanitize=True
    )


def _same(x: torch.Tensor) -> torch.Tensor:
    return x


def _scale(dtype: torch.dtype, rows: int, width: int = 1):
    """Per-token-head planes with a scale plane of this dtype, row count and width."""
    return lambda cache: dataclasses.replace(
        cache.kv_cache, scale=torch.zeros(rows, 1, width, dtype=dtype, device="cuda")
    )


@pytest.mark.parametrize(
    "change,message",
    [
        # Query, q_pe and latent: shapes, one dtype, dense channels, RoPE width.
        (dict(query=lambda r: r["query"].flatten(1)), "tokens, heads"),
        (dict(query=lambda r: r["query"].float()), "fp16 or bf16"),
        (dict(query=lambda r: r["query"].double()), "fp16 or bf16"),
        (dict(query=lambda r: r["query"].to(FP8)), "fp16 or bf16"),
        (dict(query=lambda r: r["query"].to(torch.int16)), "fp16 or bf16"),
        (
            dict(
                query=lambda r: r["query"].float(),
                override=lambda r: "triton_mla_prologue",
            ),
            "fp16 or bf16",
        ),
        (dict(latent_cache=lambda r: r["latent_cache"][:, :64]), "latent_cache"),
        (dict(latent_cache=lambda r: r["latent_cache"].unsqueeze(1)), "latent_cache"),
        (dict(latent_cache=lambda r: r["latent_cache"][:2]), "latent_cache"),
        (
            dict(latent_cache=lambda r: torch.cat([r["latent_cache"]] * 2)),
            "latent_cache",
        ),
        (dict(q_pe=lambda r: r["q_pe"].flatten(1)), "q_pe"),
        (dict(q_pe=lambda r: r["q_pe"][0, 0, 0]), "q_pe"),
        (dict(latent_cache=lambda r: r["latent_cache"][0, 0]), "latent_cache"),
        (dict(q_pe=lambda r: r["q_pe"][:2]), "q_pe"),
        (dict(q_pe=lambda r: r["q_pe"][:, :1].contiguous()), "q_pe"),
        (dict(q_pe=lambda r: r["q_pe"].half()), "share a dtype"),
        (dict(latent_cache=lambda r: r["latent_cache"].half()), "share a dtype"),
        (dict(query=lambda r: r["query"].half()), "share a dtype"),
        (dict(q_pe=lambda r: _strided(r["q_pe"])), "dense channels"),
        (dict(query=lambda r: _strided(r["query"])), "dense channels"),
        (dict(latent_cache=lambda r: _strided(r["latent_cache"])), "dense channels"),
        (
            dict(query=lambda r: r["query"][:, :1].expand(-1, 4, -1)),
            "share addresses",
        ),
        (dict(q_pe=lambda r: r["q_pe"][:, :1].expand(-1, 4, -1)), "share addresses"),
        (
            dict(
                query=lambda r: r["query"][:3, :3],
                q_pe=lambda r: r["query"][..., 512:].transpose(0, 1),
            ),
            "must be that view",
        ),
        (dict(q_pe=lambda r: r["q_pe"].new_zeros(3, 4, 65)[..., 1:]), "16-byte"),
        (
            dict(latent_cache=lambda r: r["latent_cache"].new_zeros(3, 577)[:, 1:]),
            "16-byte",
        ),
        (
            dict(latent_cache=lambda r: r["latent_cache"][:1].expand(3, -1)),
            "share addresses",
        ),
        (
            dict(
                q_pe=lambda r: r["q_pe"][..., :32].contiguous(),
                latent_cache=lambda r: r["latent_cache"][:, :544].contiguous(),
                query=lambda r: r["query"][..., :544],
                rotary=lambda r: None,
            ),
            "RoPE is 64",
        ),
        # Write slots.
        (dict(cache__slots=lambda c: _strided(c.slots)), "dense vector"),
        (dict(cache__slots=lambda c: torch.cat([c.slots, c.slots[:1]])), "cache slots"),
        (dict(cache__slots=lambda c: c.slots.float()), "dense vector"),
        # Cache rows and planes.
        (
            dict(
                cache=_plane_cache(512, 64), cache__kv_cache=_scale(torch.float, 8, 2)
            ),
            "1 scale",
        ),
        (
            dict(
                cache=_plane_cache(512, 64),
                cache__kv_cache=lambda c: dataclasses.replace(
                    c.kv_cache, scale=torch.zeros(8, 2, 1, device="cuda")
                ),
            ),
            "1 scale",
        ),
        (dict(cache__kv_cache=lambda c: _strided(c.kv_cache)), "latent cache rows"),
        (
            dict(cache__kv_cache=lambda c: c.kv_cache.expand(-1, 2, -1).contiguous()),
            "latent cache rows",
        ),
        (dict(cache__kv_cache=lambda c: c.kv_cache.squeeze(1)), "latent cache rows"),
        (
            dict(cache__kv_cache=lambda c: c.kv_cache[:1].expand(8, -1, -1)),
            "latent cache rows",
        ),
        (dict(cache__kv_cache=lambda c: c.kv_cache.unsqueeze(1)), "latent cache rows"),
        (dict(cache__kv_cache=lambda c: c.kv_cache[..., :512]), "latent cache rows"),
        (
            dict(cache__kv_cache=lambda c: torch.cat([c.kv_cache] * 2, -1)[..., :640]),
            "latent cache rows",
        ),
        (dict(cache=_plane_cache(448, 64)), "1 scale"),
        (dict(cache=_plane_cache(512, 32)), "1 scale"),
        (
            dict(
                cache=_plane_cache(512, 64),
                cache__kv_cache=lambda c: dataclasses.replace(
                    c.kv_cache, scale=c.kv_cache.scale.squeeze(1)
                ),
            ),
            "1 scale",
        ),
        (
            dict(
                cache=_plane_cache(512, 64),
                cache__kv_cache=lambda c: dataclasses.replace(
                    c.kv_cache, rope=c.kv_cache.rope[0, 0, 0]
                ),
            ),
            "1 scale",
        ),
        (
            dict(cache=_plane_cache(512, 64), cache__kv_cache=_scale(torch.half, 8)),
            "per-token-head planes",
        ),
        (
            dict(cache=_plane_cache(512, 64), cache__kv_cache=_scale(torch.float, 2)),
            "per-token-head planes",
        ),
        (
            dict(cache=_plane_cache(512, 64), cache__kv_cache=_scale(torch.float, 16)),
            "per-token-head planes",
        ),
        (
            dict(
                cache=_plane_cache(512, 64),
                cache__kv_cache=lambda c: dataclasses.replace(
                    c.kv_cache, rope=c.kv_cache.rope[:2]
                ),
            ),
            "per-token-head planes",
        ),
        (
            dict(
                cache=_plane_cache(512, 64),
                cache__kv_cache=lambda c: dataclasses.replace(
                    c.kv_cache, latent=c.kv_cache.latent[:2]
                ),
            ),
            "per-token-head planes",
        ),
        (dict(cache__kv_cache=lambda c: c.kv_cache.half()), "native cache"),
        # Rotary: no M-RoPE, dense positions, an fp32 table as wide as the RoPE.
        (dict(rotary__mrope=lambda _: MRope((16, 8, 8), False)), "multimodal"),
        (dict(rotary__positions=lambda r: r.positions[:2]), "dense rows"),
        (
            dict(rotary__positions=lambda r: torch.cat([r.positions, r.positions[:1]])),
            "dense rows",
        ),
        (dict(rotary__positions=lambda r: r.positions.float()), "dense rows"),
        (dict(rotary__cos_sin_cache=lambda r: r.cos_sin_cache.to(BF16)), "fp32 rows"),
        (dict(rotary__cos_sin_cache=lambda _: cos_sin_cache(32)), "RoPE channels"),
        (dict(rotary__cos_sin_cache=lambda _: cos_sin_cache(128)), "RoPE channels"),
        # Absorbed query width, then expanded keys and values.
        (dict(query=lambda r: r["query"][..., 256:]), "kv_lora_rank"),
        (
            dict(query=lambda r: torch.cat([r["query"], r["query"][..., :128]], -1)),
            "kv_lora_rank",
        ),
        (_expanded(lambda n: n[..., :64], _same), "expanded k_nope"),
        (_expanded(lambda n: n.half(), _same), "expanded k_nope"),
        (_expanded(_same, lambda n: n.half()), "expanded k_nope"),
        (_expanded(lambda n: n.half(), lambda n: n.half()), "expanded k_nope"),
        (_expanded(_strided, _same), "expanded k_nope"),
        (_expanded(_same, _strided), "expanded k_nope"),
        (_expanded(_same, lambda n: n[:, :2]), "expanded k_nope"),
        (_expanded(_same, lambda n: n[..., 0].contiguous()), "expanded k_nope"),
        (_expanded(_same, lambda n: n.unflatten(-1, (2, 64))), "expanded k_nope"),
        (_expanded(_same, lambda n: n[:2]), "expanded k_nope"),
        (_expanded(lambda n: n[:1].expand(3, -1, -1), _same), "expanded k_nope"),
        (_expanded(_same, lambda n: n[:1].expand(3, -1, -1)), "expanded k_nope"),
    ],
)
def test_mla_entry_rejects_malformed_requests(change, message):
    with pytest.raises(ValueError, match=message):
        mla_prologue(**_mla_request(**change))


@pytest.mark.parametrize(
    "mode,vendors",
    [
        ("gqa_prologue", {"amd", "ascend", "nvidia"}),
        ("mla_prologue", {"amd", "nvidia"}),
    ],
)
def test_every_supported_vendor_has_a_prologue(mode, vendors):
    """Qwen3 serves on Ascend; the composite is the kernel there."""
    served = set()
    for spec in KernelRegistry.get().list_kernels("attention", mode):
        served |= spec.capability.vendors
    assert vendors <= served


def test_prologue_kernels_constrain_only_traits_the_entries_state(monkeypatch):
    """A misspelled trait would silently admit every value of the trait it meant."""
    import tokenspeed_kernel.ops.attention.prologue as prologue

    stated = {}

    def spy(mode, dtype, traits, solution, override):
        stated[mode] = {key for key, _ in traits}
        raise LookupError

    monkeypatch.setattr(prologue, "_select", spy)
    with pytest.raises(LookupError):
        gqa_prologue(**_gqa_request())
    with pytest.raises(LookupError):
        mla_prologue(**_mla_request())
    for mode, keys in stated.items():
        for spec in KernelRegistry.get().list_kernels("attention", mode):
            bounds = {f"{key}{end}" for key in keys for end in ("_min", "_max")}
            assert set(spec.traits) <= keys | bounds, spec.name
