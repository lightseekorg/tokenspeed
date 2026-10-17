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

from __future__ import annotations

import contextlib

import pytest
import tokenspeed_kernel.ops.embedding.triton as embedding_triton
import torch
from tokenspeed_kernel.ops.embedding import (
    apply_k_rope,
    apply_rope,
    apply_rope_mla,
    mxfp8_embedding,
)
from tokenspeed_kernel.ops.embedding.triton import _mxfp8_embedding_kernel
from tokenspeed_kernel.platform import current_platform
from utils import assert_no_triton_compile


@pytest.mark.parametrize("solution", ["triton", "cuda"])
def test_rope_neox_full_bf16(
    device: str,
    solution: str,
    require,
) -> None:
    torch.manual_seed(0)
    num_tokens = 17
    num_q_heads = 8
    num_k_heads = 2
    head_size = 128
    rotary_dim = 128
    max_position = 1024
    dtype = torch.bfloat16
    require("embedding", "rope", solution, dtype, "q")

    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
            / rotary_dim
        )
    )
    t = torch.arange(max_position, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()

    positions = torch.randint(
        0, max_position, (num_tokens,), device=device, dtype=torch.int64
    )
    query = torch.randn(num_tokens, num_q_heads * head_size, device=device, dtype=dtype)
    key = torch.randn(num_tokens, num_k_heads * head_size, device=device, dtype=dtype)

    # Reference (PyTorch).
    cos_sin_ref = cos_sin_cache.index_select(0, positions)
    cos_ref, sin_ref = cos_sin_ref.chunk(2, dim=-1)
    cos_ref = cos_ref.unsqueeze(-2).to(dtype)
    sin_ref = sin_ref.unsqueeze(-2).to(dtype)

    q_ref = query.clone().view(num_tokens, num_q_heads, head_size)
    q1, q2 = torch.chunk(q_ref, 2, dim=-1)
    q_out = torch.cat(
        (q1 * cos_ref - q2 * sin_ref, q2 * cos_ref + q1 * sin_ref), dim=-1
    )
    q_ref = q_out.reshape(num_tokens, num_q_heads * head_size)

    k_ref = key.clone().view(num_tokens, num_k_heads, head_size)
    k1, k2 = torch.chunk(k_ref, 2, dim=-1)
    k_out = torch.cat(
        (k1 * cos_ref - k2 * sin_ref, k2 * cos_ref + k1 * sin_ref), dim=-1
    )
    k_ref = k_out.reshape(num_tokens, num_k_heads * head_size)

    apply_rope(
        positions=positions,
        q=query,
        k=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=True,
        solution=solution,
    )

    torch.testing.assert_close(query, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(key, k_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("solution", ["triton", "cuda"])
def test_rope_gptj_full_bf16(
    device: str,
    solution: str,
    require,
) -> None:
    torch.manual_seed(1)
    num_tokens = 9
    num_q_heads = 4
    num_k_heads = 2
    head_size = 64
    rotary_dim = 64
    max_position = 512
    dtype = torch.bfloat16
    require("embedding", "rope", solution, dtype, "q")

    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
            / rotary_dim
        )
    )
    t = torch.arange(max_position, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()

    positions = torch.randint(
        0, max_position, (num_tokens,), device=device, dtype=torch.int64
    )
    query = torch.randn(num_tokens, num_q_heads * head_size, device=device, dtype=dtype)
    key = torch.randn(num_tokens, num_k_heads * head_size, device=device, dtype=dtype)

    cos_sin_ref = cos_sin_cache.index_select(0, positions)
    cos_ref, sin_ref = cos_sin_ref.chunk(2, dim=-1)
    cos_ref = cos_ref.unsqueeze(-2).to(dtype)
    sin_ref = sin_ref.unsqueeze(-2).to(dtype)

    def _gptj_ref(x, num_heads):
        x = x.view(num_tokens, num_heads, head_size)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        o1 = x1 * cos_ref - x2 * sin_ref
        o2 = x2 * cos_ref + x1 * sin_ref
        return (
            torch.stack((o1, o2), dim=-1)
            .flatten(-2)
            .reshape(num_tokens, num_heads * head_size)
        )

    q_ref = _gptj_ref(query.clone(), num_q_heads)
    k_ref = _gptj_ref(key.clone(), num_k_heads)

    apply_rope(
        positions=positions,
        q=query,
        k=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=False,
        solution=solution,
    )

    torch.testing.assert_close(query, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(key, k_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("solution", ["triton", "cuda"])
def test_rope_neox_partial_bf16(
    device: str,
    solution: str,
    require,
) -> None:
    torch.manual_seed(2)
    num_tokens = 5
    num_q_heads = 4
    num_k_heads = 1
    head_size = 128
    rotary_dim = 64
    max_position = 256
    dtype = torch.bfloat16
    require("embedding", "rope", solution, dtype, "q")

    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
            / rotary_dim
        )
    )
    t = torch.arange(max_position, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()

    positions = torch.randint(
        0, max_position, (num_tokens,), device=device, dtype=torch.int64
    )
    query = torch.randn(num_tokens, num_q_heads * head_size, device=device, dtype=dtype)
    key = torch.randn(num_tokens, num_k_heads * head_size, device=device, dtype=dtype)
    query_orig = query.clone()
    key_orig = key.clone()

    cos_sin_ref = cos_sin_cache.index_select(0, positions)
    cos_ref, sin_ref = cos_sin_ref.chunk(2, dim=-1)
    cos_ref = cos_ref.unsqueeze(-2).to(dtype)
    sin_ref = sin_ref.unsqueeze(-2).to(dtype)

    def _ref(x, num_heads):
        x = x.view(num_tokens, num_heads, head_size)
        rot = x[..., :rotary_dim]
        rest = x[..., rotary_dim:]
        r1, r2 = torch.chunk(rot, 2, dim=-1)
        rot_out = torch.cat(
            (r1 * cos_ref - r2 * sin_ref, r2 * cos_ref + r1 * sin_ref), dim=-1
        )
        return torch.cat((rot_out, rest), dim=-1).reshape(
            num_tokens, num_heads * head_size
        )

    q_ref = _ref(query.clone(), num_q_heads)
    k_ref = _ref(key.clone(), num_k_heads)

    apply_rope(
        positions=positions,
        q=query,
        k=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=True,
        solution=solution,
    )

    torch.testing.assert_close(query, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(key, k_ref, rtol=2e-2, atol=2e-2)

    q_view = query.view(num_tokens, num_q_heads, head_size)
    q_orig_view = query_orig.view(num_tokens, num_q_heads, head_size)
    assert torch.equal(q_view[..., rotary_dim:], q_orig_view[..., rotary_dim:])
    k_view = key.view(num_tokens, num_k_heads, head_size)
    k_orig_view = key_orig.view(num_tokens, num_k_heads, head_size)
    assert torch.equal(k_view[..., rotary_dim:], k_orig_view[..., rotary_dim:])


@pytest.mark.parametrize("solution", ["triton", "cuda"])
def test_rope_single_token(
    device: str,
    solution: str,
    require,
) -> None:
    """Edge case: num_tokens == 1 (decode step)."""
    torch.manual_seed(4)
    num_tokens = 1
    num_q_heads = 8
    num_k_heads = 1
    head_size = 128
    rotary_dim = 128
    max_position = 64
    dtype = torch.bfloat16
    require("embedding", "rope", solution, dtype, "q")

    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
            / rotary_dim
        )
    )
    t = torch.arange(max_position, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()

    positions = torch.tensor([5], device=device, dtype=torch.int64)
    query = torch.randn(num_tokens, num_q_heads * head_size, device=device, dtype=dtype)
    key = torch.randn(num_tokens, num_k_heads * head_size, device=device, dtype=dtype)

    cos_sin_ref = cos_sin_cache.index_select(0, positions)
    cos_ref, sin_ref = cos_sin_ref.chunk(2, dim=-1)
    cos_ref = cos_ref.unsqueeze(-2).to(dtype)
    sin_ref = sin_ref.unsqueeze(-2).to(dtype)

    def _ref(x, num_heads):
        x = x.view(num_tokens, num_heads, head_size)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat(
            (x1 * cos_ref - x2 * sin_ref, x2 * cos_ref + x1 * sin_ref), dim=-1
        ).reshape(num_tokens, num_heads * head_size)

    q_ref = _ref(query.clone(), num_q_heads)
    k_ref = _ref(key.clone(), num_k_heads)

    apply_rope(
        positions=positions,
        q=query,
        k=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=True,
        solution=solution,
    )

    torch.testing.assert_close(query, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(key, k_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("solution", [None, "triton", "flashinfer"])
@pytest.mark.parametrize("is_neox", [True, False])
def test_rope_mla_quantize(
    device: str,
    solution: str,
    is_neox: bool,
    require,
) -> None:
    torch.manual_seed(6)
    dtype = torch.bfloat16
    if solution is not None:
        require("embedding", "rope_mla", solution, dtype, "q_rope")

    num_tokens = 13
    num_heads = 4
    nope_dim = 32
    rope_dim = 64
    max_position = 512
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim)
    )
    t = torch.arange(max_position, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()

    positions = torch.randint(
        0, max_position, (num_tokens,), device=device, dtype=torch.int64
    )
    cos, sin = cos_sin_cache.index_select(0, positions).chunk(2, dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rope_ref = lambda x: (
        torch.cat(
            (
                torch.chunk(x.float(), 2, dim=-1)[0] * cos
                - torch.chunk(x.float(), 2, dim=-1)[1] * sin,
                torch.chunk(x.float(), 2, dim=-1)[1] * cos
                + torch.chunk(x.float(), 2, dim=-1)[0] * sin,
            ),
            dim=-1,
        )
        if is_neox
        else torch.stack(
            (
                x.float()[..., ::2] * cos - x.float()[..., 1::2] * sin,
                x.float()[..., 1::2] * cos + x.float()[..., ::2] * sin,
            ),
            dim=-1,
        ).flatten(-2)
    ).to(x.dtype)
    q_rope = torch.randn(num_tokens, num_heads, rope_dim, device=device, dtype=dtype)
    k_rope = torch.randn(num_tokens, num_heads, rope_dim, device=device, dtype=dtype)
    q_nope = torch.randn(num_tokens, num_heads, nope_dim, device=device, dtype=dtype)
    k_nope = torch.randn(num_tokens, num_heads, nope_dim, device=device, dtype=dtype)

    quant_scale_q = 2.0
    quant_scale_kv = 2.0

    query_fp8, key_fp8 = apply_rope_mla(
        positions=positions,
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        quant_scale_q=quant_scale_q,
        quant_scale_kv=quant_scale_kv,
        solution=solution,
    )

    q_rope_ref = rope_ref(q_rope)
    k_rope_ref = rope_ref(k_rope)
    q_ref = torch.cat(
        (q_nope.float() * quant_scale_q, q_rope_ref.float() * quant_scale_q),
        dim=-1,
    ).to(torch.float8_e4m3fn)
    k_ref = torch.cat(
        (k_nope.float() * quant_scale_kv, k_rope_ref.float() * quant_scale_kv),
        dim=-1,
    ).to(torch.float8_e4m3fn)

    assert query_fp8.shape == q_ref.shape
    assert key_fp8.shape == k_ref.shape
    assert query_fp8.dtype == torch.float8_e4m3fn
    assert key_fp8.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(query_fp8.float(), q_ref.float(), rtol=0, atol=0.5)
    torch.testing.assert_close(key_fp8.float(), k_ref.float(), rtol=0, atol=0.5)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scales", ["unit", "float", "tensor"])
@pytest.mark.parametrize(
    "layout, num_heads, nope_dim, rope_dim, token_counts",
    [
        # forward_normal_chunked_kv_prepare (Kimi-K3 prefill): per-head K from
        # kv_b_proj, the latent's single RoPE row expanded over the heads.
        ("expanded", 96, 128, 64, (1, 3, 128, 4095, 4096)),
        # The absorbed FP8-cache prologue: one latent K row per token, also
        # with a single query head (one head per program).
        ("absorbed", 6, 512, 64, (1, 3, 128, 4095, 4096)),
        ("absorbed", 1, 512, 64, (1, 3, 128, 4095, 4096)),
        # Per-head k_nope with a single-head k_rope, and per-head K throughout,
        # also with feature widths that are not powers of two.
        ("single_k_rope", 20, 128, 64, (1, 3, 128, 4095, 4096)),
        ("single_k_rope", 7, 72, 24, (1, 3, 128, 4095, 4096)),
        ("per_head", 3, 64, 32, (1, 3, 128, 4095, 4096)),
        ("per_head", 5, 96, 40, (1, 3, 128, 4095, 4096)),
        # RoPE rows far wider than the NoPE rows: the RoPE tile bounds the
        # heads per program (128 heads of 16384 columns exceed Triton's
        # largest tile).
        ("per_head", 65, 16, 8200, (1, 3, 128)),
    ],
)
def test_nope_mla_quantize_is_exact_for_any_token_count(
    device: str,
    layout: str,
    num_heads: int,
    nope_dim: int,
    rope_dim: int,
    token_counts: tuple[int, ...],
    scales: str,
    dtype: torch.dtype,
    require,
) -> None:
    """Every output byte is ``(x.float() * scale).clamp(-448, 448)`` cast to
    e4m3, for any token count and without a recompile; nothing outside the
    output views is written."""
    require("embedding", "rope_mla", "triton", dtype, "q_rope")
    rank, pad = 512, 16
    kv_heads = 1 if layout == "absorbed" else num_heads
    scale_q, scale_kv = {
        "unit": (1.0, 1.0),
        "float": (0.37, 3.0),
        "tensor": (
            torch.tensor([0.5], device=device),
            torch.tensor([6.5], device=device),
        ),
    }[scales]
    generator = torch.Generator(device=device).manual_seed(5)

    for i, num_tokens in enumerate(token_counts):
        # The compile key holds no token count: the variants the first count
        # compiles (fresh and strided outputs) serve every later count.
        with (
            assert_no_triton_compile(embedding_triton._mla_nope_quantize_fp8_kernel)
            if i
            else contextlib.nullcontext()
        ):
            # |x| <= 64 keeps every scaled value inside e4m3's range.
            q, kv, latent, k = (
                torch.randn(shape, device=device, generator=generator)
                .mul_(16)
                .clamp_(-64, 64)
                .to(dtype)
                for shape in (
                    (num_tokens, num_heads, nope_dim + rope_dim),
                    (num_tokens, num_heads, 2 * nope_dim),
                    (num_tokens, 1, rank + rope_dim),
                    (num_tokens, num_heads, nope_dim + rope_dim),
                )
            )
            q_nope, q_rope = q[..., :nope_dim], q[..., nope_dim:]
            if layout == "expanded":
                k_nope = kv[..., :nope_dim]
                k_rope = latent[..., rank:].expand(-1, num_heads, -1)
            elif layout == "absorbed":
                k_nope, k_rope = latent[..., :nope_dim], latent[..., rank:]
            else:
                k_nope = k[..., :nope_dim]
                k_rope = (
                    k[:, :1, nope_dim:]
                    if layout == "single_k_rope"
                    else k[..., nope_dim:]
                )
            if current_platform().is_nvidia:
                # Past e4m3's range NVIDIA's cvt.rn.satfinite saturates to +-448.
                specials = torch.tensor(
                    [1e4, -1e4, float("inf"), -float("inf")], device=device
                )
                q[0, :, :4] = specials
                k_nope[0, :, :4] = specials

            query, key = apply_rope_mla(
                positions=None,
                q_rope=q_rope,
                k_rope=k_rope,
                q_nope=q_nope,
                k_nope=k_nope,
                cos_sin_cache=None,
                is_neox=False,
                quant_scale_q=scale_q,
                quant_scale_kv=scale_kv,
                solution="triton",
            )
            # Strided outputs inside NaN-filled buffers (0x7F) with a spare head
            # and spare columns, which must come back untouched.
            query_buf, key_buf = (
                torch.full(
                    (num_tokens, heads + 1, nope_dim + rope_dim + pad),
                    0x7F,
                    dtype=torch.uint8,
                    device=device,
                )
                for heads in (num_heads, kv_heads)
            )
            query_out, key_out = (
                buf.view(torch.float8_e4m3fn)[:, :heads]
                for buf, heads in ((query_buf, num_heads), (key_buf, kv_heads))
            )
            apply_rope_mla(
                positions=None,
                q_rope=q_rope,
                k_rope=k_rope,
                q_nope=q_nope,
                k_nope=k_nope,
                cos_sin_cache=None,
                is_neox=False,
                quant_scale_q=scale_q,
                quant_scale_kv=scale_kv,
                q_nope_out=query_out[..., :nope_dim],
                q_rope_out=query_out[..., nope_dim : nope_dim + rope_dim],
                k_nope_out=key_out[..., :nope_dim],
                k_rope_out=key_out[..., nope_dim : nope_dim + rope_dim],
                solution="triton",
            )

            want_query, want_key = (
                (
                    torch.cat(
                        (nope, rope.expand(-1, nope.shape[1], -1)), dim=-1
                    ).float()
                    * torch.as_tensor(scale, dtype=torch.float32, device=device)
                )
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
                for nope, rope, scale in (
                    (q_nope, q_rope, scale_q),
                    (k_nope, k_rope, scale_kv),
                )
            )
            for got, want in (
                (query, want_query),
                (key, want_key),
                (query_out[..., : nope_dim + rope_dim], want_query),
                (key_out[..., : nope_dim + rope_dim], want_key),
            ):
                assert torch.equal(got.view(torch.uint8), want)
            for buf, heads in ((query_buf, num_heads), (key_buf, kv_heads)):
                assert bool((buf[:, heads:] == 0x7F).all())
                assert bool((buf[..., -pad:] == 0x7F).all())


@pytest.mark.parametrize("solution", ["triton", "cuda"])
@pytest.mark.parametrize("is_neox", [True, False])
def test_k_only_rope_matches_a_paired_call(
    device: str, solution: str, is_neox: bool, require
) -> None:
    """Dropping the scratch query must not change what the key becomes."""
    torch.manual_seed(0)
    num_tokens, head_size, max_position = 13, 64, 512
    dtype = torch.bfloat16
    require("embedding", "rope", solution, dtype, "q")

    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, head_size, 2, device=device, dtype=torch.float32)
            / head_size
        )
    )
    freqs = torch.outer(
        torch.arange(max_position, device=device, dtype=torch.float32), inv_freq
    )
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()
    positions = torch.randint(max_position, (num_tokens,), device=device)
    key = torch.randn(num_tokens, head_size, device=device, dtype=dtype)

    paired = key.clone()
    apply_rope(
        positions,
        paired.new_empty(paired.shape),
        paired,
        head_size,
        cos_sin_cache,
        is_neox=is_neox,
        solution=solution,
    )
    k_only = key.clone()
    apply_k_rope(
        positions,
        k_only,
        head_size,
        cos_sin_cache,
        is_neox=is_neox,
        solution=solution,
    )
    torch.testing.assert_close(k_only, paired, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_vocab_shard_embedding_masks_other_shards(
    dtype: torch.dtype, index_dtype: torch.dtype, device: str
) -> None:
    """One gather reproduces the mask-then-lookup vocabulary-parallel path."""
    from tokenspeed_kernel.ops.embedding import vocab_shard_embedding

    torch.manual_seed(3)
    org_start, org_end, padding, added_start, added_end = 500, 1000, 24, 2000, 2016
    weight = torch.randn(
        org_end - org_start + padding + added_end - added_start + 8,
        96,
        device=device,
        dtype=dtype,
    )
    ids = torch.tensor(
        [0, 499, 500, 777, 999, 1000, 1999, 2000, 2015, 2016, 5000],
        device=device,
        dtype=index_dtype,
    ).view(1, -1)

    original = (ids >= org_start) & (ids < org_end)
    added = (ids >= added_start) & (ids < added_end)
    added_offset = added_start - (org_end - org_start) - padding
    local = (original | added) * (ids - org_start * original - added_offset * added)
    expected = torch.nn.functional.embedding(local, weight)
    expected.masked_fill_(~(original | added).unsqueeze(-1), 0)

    out = vocab_shard_embedding(
        weight, ids, (org_start, org_end), padding, (added_start, added_end)
    )
    assert out.shape == (1, ids.shape[1], 96) and out.dtype == dtype
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert out[0, [0, 1, 5, 6, 9, 10]].eq(0).all()
    assert vocab_shard_embedding(
        weight, ids[:, :0], (org_start, org_end), padding, (added_start, added_end)
    ).shape == (1, 0, 96)
    with pytest.raises(ValueError):
        vocab_shard_embedding(
            weight, ids, (org_start, org_end), padding, (added_start, added_end + 100)
        )
    with pytest.raises(ValueError):
        vocab_shard_embedding(
            weight, ids[:, ::2], (org_start, org_end), padding, (added_start, added_end)
        )


def test_only_the_cuda_rope_stores_kv_in_its_launch(device: str) -> None:
    """The Triton rope declines a fused K/V write, and refuses one it is forced to take."""
    from tokenspeed_kernel.ops.embedding.triton import triton_embedding_rope
    from tokenspeed_kernel.registry import KernelRegistry

    for spec in KernelRegistry.get().list_kernels("embedding", "rope"):
        if spec.solution == "triton":
            assert spec.traits["has_fused_kv"] == frozenset({False})
    q = torch.zeros(1, 64, dtype=torch.bfloat16, device=device)
    table = torch.zeros(4, 64, device=device)
    with pytest.raises(ValueError, match="fused KV"):
        triton_embedding_rope(
            positions=torch.zeros(1, dtype=torch.int64, device=device),
            q=q,
            k=q.clone(),
            head_size=64,
            cos_sin_cache=table,
            fused_set_kv_buffer_arg=object(),
        )


def test_triton_rope_reads_int32_positions_past_two_to_the_25():
    """An int32 position times the table row stride passes 2^31 at 2^25 rows."""
    position, head_size = 2**25, 64
    table = torch.zeros(position + 1, head_size, device="cuda", dtype=torch.float32)
    table[position, : head_size // 2] = 0.6
    table[position, head_size // 2 :] = 0.8
    outs = []
    for dtype in (torch.int64, torch.int32):
        g = torch.Generator(device="cuda").manual_seed(7)
        q = torch.randn(
            1, 2 * head_size, device="cuda", dtype=torch.bfloat16, generator=g
        )
        k = torch.randn(1, head_size, device="cuda", dtype=torch.bfloat16, generator=g)
        apply_rope(
            positions=torch.tensor([position], device="cuda", dtype=dtype),
            q=q,
            k=k,
            head_size=head_size,
            cos_sin_cache=table,
            is_neox=True,
            solution="triton",
        )
        outs.append((q, k))
    for a, b in zip(*outs):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def _mxfp8_table(device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(83)
    weight = torch.randn((64, 256), device=device, generator=generator).to(
        torch.float8_e4m3fn
    )
    scales = torch.randint(
        118, 132, (64, 8), device=device, dtype=torch.uint8, generator=generator
    )
    return weight, scales


def _mxfp8_indices(device, rows: int, layout: str, dtype: torch.dtype) -> torch.Tensor:
    if layout == "layer":
        ids = torch.arange(rows * 2 * 24, device=device, dtype=dtype).reshape(
            rows, 2, 24
        )[:, 1, :]
    elif layout == "transpose":
        ids = torch.arange(rows * 24, device=device, dtype=dtype).reshape(24, rows).t()
    elif layout == "columns":
        ids = torch.arange(rows * 48, device=device, dtype=dtype).reshape(rows, 48)[
            :, ::2
        ]
    elif layout == "broadcast":
        return (torch.arange(24, device=device, dtype=dtype) * 4).expand(rows, -1)
    else:
        ids = torch.arange(rows * 24, device=device, dtype=dtype).reshape(rows, 2, 12)
    # Update in place to retain the sliced/strided layout.
    ids.copy_(ids % 90)
    return ids


def _mxfp8_reference(weight, scales, ids):
    local = (ids >= 17) & (ids < 70)
    offset = (ids.long() - 17).clamp(0, 63)
    codes = weight.view(torch.uint8)[offset].view(torch.float8_e4m3fn).float()
    factors = (
        scales[offset].view(torch.float8_e8m0fnu).float().repeat_interleave(32, dim=-1)
    )
    return (codes * factors).to(torch.bfloat16).masked_fill(~local[..., None], 0)


@pytest.mark.parametrize("rows", [0, 6, 12, 24, 48, 96, 192, 1031])
@pytest.mark.parametrize(
    "layout", ["layer", "transpose", "columns", "broadcast", "contiguous"]
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_embedding_reads_strided_ids(device, rows, layout, dtype):
    weight, scales = _mxfp8_table(device)
    ids = _mxfp8_indices(device, rows, layout, dtype)
    actual = mxfp8_embedding(weight, scales, ids, 17, 70)
    expected = _mxfp8_reference(weight, scales, ids)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous()


def test_embedding_rejects_noncontiguous_three_dimensional_ids(device):
    weight, scales = _mxfp8_table(device)
    ids = torch.zeros((2, 3, 4), device=device, dtype=torch.int64).transpose(0, 1)
    with pytest.raises(ValueError, match="contiguous or two-dimensional IDs"):
        mxfp8_embedding(weight, scales, ids, 17, 70)


def test_embedding_replays_updated_layer_slice(device):
    weight, scales = _mxfp8_table(device)
    ids = _mxfp8_indices(device, 6, "layer", torch.int64)
    mxfp8_embedding(weight, scales, ids, 17, 70)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mxfp8_embedding(weight, scales, ids, 17, 70)
    for shift in (1, 3, 11):
        ids.copy_((ids + shift) % 90)
        graph.replay()
        torch.testing.assert_close(
            captured, _mxfp8_reference(weight, scales, ids), rtol=0, atol=0
        )


def test_embedding_id_strides_do_not_specialize(device):
    weight, scales = _mxfp8_table(device)
    ids = torch.arange(48, device=device, dtype=torch.int64).reshape(2, 24)
    storage = torch.zeros((12, 3, 48), device=device, dtype=torch.int64)
    first = storage[:2, 1, :24]
    first.copy_(ids)
    mxfp8_embedding(weight, scales, first, 17, 70)
    second = storage[:, 2, ::2].t()
    with assert_no_triton_compile(_mxfp8_embedding_kernel):
        actual = mxfp8_embedding(weight, scales, second, 17, 70)
    torch.testing.assert_close(
        actual, _mxfp8_reference(weight, scales, second), rtol=0, atol=0
    )


@pytest.mark.parametrize("rows", [2048, 2049, 65536])
def test_embedding_strided_large_row_dispatch(device, rows):
    weight, scales = _mxfp8_table(device)
    ids = torch.arange(rows * 2, device=device, dtype=torch.int64).reshape(rows, 2)[
        :, 1:2
    ]
    ids.copy_(ids % 90)
    actual = mxfp8_embedding(weight, scales, ids, 17, 70)
    torch.testing.assert_close(
        actual, _mxfp8_reference(weight, scales, ids), rtol=0, atol=0
    )
