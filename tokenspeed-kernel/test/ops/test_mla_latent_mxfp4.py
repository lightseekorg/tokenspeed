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

"""MXFP4 MLA latent rows: the store's bytes against an independent codec, the
gather's decode, the MLA prologue's write, and sparse MLA over the packed rows
against an FP32 reference."""

from __future__ import annotations

import pytest
import torch
from attention_prologue_helpers import latent_target, mla_inputs, mla_query
from tokenspeed_kernel.ops.attention.dsa.mxfp4 import mxfp4_sparse_mla_attention
from tokenspeed_kernel.ops.attention.prologue import (
    KVCacheFormat,
    mla_prologue,
)
from tokenspeed_kernel.ops.kvcache.mla_mxfp4 import (
    dequant_gather_mla_latent_mxfp4,
    mla_mxfp4_row_bytes,
    quantize_store_mla_latent_mxfp4,
)
from tokenspeed_kernel.platform import current_platform

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

RANK, ROPE = 512, 64
ROW = mla_mxfp4_row_bytes(RANK, ROPE)
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FLT_MIN = torch.finfo(torch.float32).tiny
_BF16_MAX = torch.finfo(torch.bfloat16).max
_QK_FORMATS = [
    "bf16",
    pytest.param(
        "mxfp8",
        marks=pytest.mark.skipif(
            not current_platform().is_cdna4, reason="block-scaled MFMA is gfx950's"
        ),
    ),
]


def _pow2_exponent_at_least(x: torch.Tensor) -> torch.Tensor:
    """Biased exponent of the smallest power of two >= x (x > 0, FP32)."""
    mantissa, exponent = torch.frexp(x)
    return exponent - 1 + (mantissa != 0.5).to(exponent.dtype) + 127


def _encode_e2m1(y: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 code, ties to the even code, sign in bit 3."""
    table = torch.tensor(_E2M1, device=y.device)
    dist = (y.abs().clamp(max=6.0)[..., None] - table).abs()
    nearest = dist == dist.amin(-1, keepdim=True)
    even = torch.arange(8, device=y.device) % 2 == 0
    # Two nearest codes are neighbours, one of them even.
    code = torch.where(
        nearest.sum(-1) > 1,
        (nearest & even).to(torch.int32).argmax(-1),
        nearest.to(torch.int32).argmax(-1),
    )
    return code | (torch.signbit(y).to(torch.int32) << 3)


def _reference_rows(latent: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """The uint8 MXFP4 rows of ``latent`` and ``rope``, by the OCP MX rule."""
    n = latent.shape[0]
    groups = latent.float().view(n, RANK // 32, 32)
    amax = groups.abs().amax(-1).clamp(min=6.0 * _FLT_MIN)
    exponent = _pow2_exponent_at_least(amax / 6.0).clamp(max=254)
    scale = torch.ldexp(torch.ones_like(amax), exponent - 127)
    codes = _encode_e2m1(groups / scale[..., None]).view(n, RANK)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    rope_bytes = rope.to(torch.bfloat16).contiguous().view(torch.uint8)
    return torch.cat([packed, exponent.to(torch.uint8), rope_bytes], dim=1)


def _decode_rows(rows: torch.Tensor) -> torch.Tensor:
    """FP32 ``[n, RANK + ROPE]`` values of uint8 MXFP4 rows."""
    n = rows.shape[0]
    packed = rows[:, : RANK // 2].to(torch.int32)
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).view(n, RANK)
    table = torch.tensor(_E2M1, device=rows.device)
    values = table[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    exponent = rows[:, RANK // 2 : RANK // 2 + RANK // 32].to(torch.int32)
    scale = torch.ldexp(torch.ones(exponent.shape, device=rows.device), exponent - 127)
    latent = (values.view(n, -1, 32) * scale[..., None]).view(n, RANK)
    rope = rows[:, RANK // 2 + RANK // 32 :].contiguous().view(torch.bfloat16)
    return torch.cat([latent, rope.float()], dim=1)


def _cache(slots: int, paged: bool) -> torch.Tensor:
    """A zeroed uint8 cache view: ``[slots, 1, ROW]``, or pages of 64 rows
    whose page stride is padded past the rows, as an arena field's is."""
    if not paged:
        return torch.zeros(slots, 1, ROW, dtype=torch.uint8, device="cuda")
    pages = -(-slots // 64)
    backing = torch.zeros(pages, 64 * ROW + 256, dtype=torch.uint8, device="cuda")
    return backing[:, : 64 * ROW].view(pages, 64, 1, ROW)


def _rows(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    if cache.dim() == 3:
        return cache[slots, 0]
    return cache[slots // 64, slots % 64, 0]


def _latent(tokens: int, seed: int, dtype=torch.bfloat16) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens, RANK + ROPE, device="cuda", generator=g)
    # Groups across a wide range of magnitudes, a zero group and exact ties.
    x[:, :RANK] *= torch.logspace(-30, 30, RANK // 32, base=2.0, device="cuda")[
        None
    ].repeat_interleave(32, dim=1)
    x[0, :32] = 0
    x[0, 32:64] = torch.tensor(
        [6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] * 4, device="cuda"
    )
    return x.to(dtype)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("tokens", [1, 37])
def test_store_writes_the_reference_codec_bytes(paged, tokens):
    latent = _latent(tokens, seed=1)
    cache = _cache(256, paged)
    slots = torch.randperm(256, device="cuda")[:tokens]
    quantize_store_mla_latent_mxfp4(
        latent[:, :RANK],
        latent[:, RANK:],
        cache,
        slots,
        write_mask=None,
        sanitize=False,
    )
    expected = _reference_rows(latent[:, :RANK], latent[:, RANK:])
    assert torch.equal(_rows(cache, slots), expected)
    untouched = torch.ones(256, dtype=torch.bool, device="cuda")
    untouched[slots] = False
    assert not _rows(cache, untouched.nonzero().squeeze(1)).any()


def test_store_writes_only_masked_in_leading_rows():
    tokens, written = 9, 6
    latent = _latent(tokens, seed=2)
    cache = _cache(64, paged=False)
    slots = torch.arange(10, 10 + written, device="cuda")
    mask = torch.tensor([1, 0, 1, 1, 0, 1], dtype=torch.bool, device="cuda")
    quantize_store_mla_latent_mxfp4(
        latent[:, :RANK],
        latent[:, RANK:],
        cache,
        slots,
        write_mask=mask,
        sanitize=False,
    )
    expected = _reference_rows(latent[:written, :RANK], latent[:written, RANK:])
    assert torch.equal(_rows(cache, slots[mask]), expected[mask])
    assert not _rows(cache, slots[~mask]).any()
    assert int(cache.count_nonzero()) == int(expected[mask].count_nonzero())


def test_sanitized_store_writes_nan_as_zero_and_inf_as_bf16_max():
    latent = _latent(2, seed=3, dtype=torch.float32)
    latent[0, 100] = float("nan")
    latent[0, 200] = float("inf")
    latent[1, RANK + 3] = float("-inf")
    latent[1, RANK + 4] = float("nan")
    cache = _cache(8, paged=True)
    slots = torch.tensor([5, 2], device="cuda")
    quantize_store_mla_latent_mxfp4(
        latent[:, :RANK],
        latent[:, RANK:],
        cache,
        slots,
        write_mask=None,
        sanitize=True,
    )
    clean = latent.nan_to_num(nan=0.0, posinf=_BF16_MAX, neginf=-_BF16_MAX)
    assert torch.equal(
        _rows(cache, slots), _reference_rows(clean[:, :RANK], clean[:, RANK:])
    )


@pytest.mark.parametrize("paged", [False, True])
def test_gather_decodes_the_rows_and_zeroes_negative_slots(paged):
    latent = _latent(20, seed=4)
    cache = _cache(128, paged)
    slots = torch.randperm(128, device="cuda")[:20]
    quantize_store_mla_latent_mxfp4(
        latent[:, :RANK],
        latent[:, RANK:],
        cache,
        slots,
        write_mask=None,
        sanitize=False,
    )
    read = torch.cat([slots[:7], torch.tensor([-1], device="cuda"), slots[7:]])
    out = torch.full((21, RANK + ROPE), float("nan"), device="cuda")
    dequant_gather_mla_latent_mxfp4(cache, read, out, kv_lora_rank=RANK)
    expected = _decode_rows(_reference_rows(latent[:, :RANK], latent[:, RANK:]))
    assert torch.equal(out[:7], expected[:7])
    assert torch.equal(out[8:], expected[7:])
    assert not out[7].any()


def test_prologue_stores_mxfp4_rows_and_assembles_the_query():
    tokens, heads = 7, 12
    q_nope, q_pe, latent = mla_inputs(tokens, heads, RANK, ROPE, seed=5)
    cache = _cache(64, paged=True)
    slots = torch.randperm(64, device="cuda")[: tokens - 2]
    target = latent_target(cache, slots, sanitize=True)
    assert target.format is KVCacheFormat.MXFP4
    out = mla_prologue(
        mla_query(q_nope, ROPE),
        q_pe,
        latent.clone(),
        expanded=None,
        rotary=None,
        cache=target,
        solution=None,
        override=None,
    )
    # Every row's query, the leading ones' latent rows.
    assert torch.equal(out.query[..., :RANK], q_nope)
    assert torch.equal(out.query[..., RANK:], q_pe)
    written = tokens - 2
    assert torch.equal(
        _rows(cache, slots),
        _reference_rows(latent[:written, :RANK], latent[:written, RANK:]),
    )


def _reference_attention(q, rows, topk_slots, topk_lens, scale):
    """FP32 sparse MLA of ``q`` over the decoded ``rows`` (slot-indexed)."""
    kv = _decode_rows(rows)
    out = torch.zeros(q.shape[0], q.shape[1], RANK, device=q.device)
    for t in range(q.shape[0]):
        selected = topk_slots[t, : int(topk_lens[t])]
        selected = selected[selected >= 0]
        keys = kv[selected]
        probs = torch.softmax(q[t].float() @ keys.T * scale, dim=-1)
        out[t] = probs @ keys[:, :RANK]
    return out


@pytest.mark.parametrize("qk_format", _QK_FORMATS)
@pytest.mark.parametrize(
    "tokens,heads,topk",
    [(1, 12, 2048), (5, 12, 256), (64, 16, 64), (3, 20, 96)],
)
@pytest.mark.parametrize("paged", [False, True])
def test_sparse_attention_matches_fp32_over_the_decoded_rows(
    qk_format, tokens, heads, topk, paged
):
    num_slots = 4096
    g = torch.Generator(device="cuda").manual_seed(6)
    latent = torch.randn(num_slots, RANK + ROPE, device="cuda", generator=g)
    cache = _cache(num_slots, paged)
    all_slots = torch.arange(num_slots, device="cuda")
    quantize_store_mla_latent_mxfp4(
        latent[:, :RANK],
        latent[:, RANK:],
        cache,
        all_slots,
        write_mask=None,
        sanitize=False,
    )
    q = torch.randn(
        tokens, heads, RANK + ROPE, device="cuda", generator=g, dtype=torch.bfloat16
    )
    topk_slots = torch.stack(
        [
            torch.randperm(num_slots, device="cuda", generator=g)[:topk]
            for _ in range(tokens)
        ]
    ).to(torch.int32)
    # Short rows, and -1 padding inside a row's live entries.
    topk_lens = torch.full((tokens,), topk, dtype=torch.int32, device="cuda")
    topk_lens[0] = max(1, topk // 3)
    topk_slots[-1, 1::7] = -1
    scale = (RANK + ROPE) ** -0.5
    out = mxfp4_sparse_mla_attention(
        q,
        cache,
        topk_slots,
        topk_lens,
        kv_lora_rank=RANK,
        softmax_scale=scale,
        qk_format=qk_format,
    )
    ref = _reference_attention(q, _rows(cache, all_slots), topk_slots, topk_lens, scale)
    assert out.shape == (tokens, heads, RANK) and out.dtype is torch.bfloat16
    # bf16: BF16 rounding of the output and P only; mxfp8 adds the query's
    # E4M3 rounding to the scores.
    tol = 2e-2 if qk_format == "bf16" else 6e-2
    err = (out.float() - ref).abs().max().item()
    assert err <= tol * ref.abs().max().item(), err


def test_sparse_attention_writes_into_a_strided_out():
    num_slots, tokens, heads = 256, 4, 12
    g = torch.Generator(device="cuda").manual_seed(7)
    latent = torch.randn(num_slots, RANK + ROPE, device="cuda", generator=g)
    cache = _cache(num_slots, paged=True)
    quantize_store_mla_latent_mxfp4(
        latent[:, :RANK],
        latent[:, RANK:],
        cache,
        torch.arange(num_slots, device="cuda"),
        write_mask=None,
        sanitize=False,
    )
    q = torch.randn(
        tokens, heads, RANK + ROPE, device="cuda", generator=g, dtype=torch.bfloat16
    )
    topk_slots = torch.randint(
        0, num_slots, (tokens, 64), device="cuda", generator=g, dtype=torch.int32
    )
    kwargs = dict(kv_lora_rank=RANK, softmax_scale=0.05, qk_format="bf16")
    expected = mxfp4_sparse_mla_attention(q, cache, topk_slots, None, **kwargs)
    backing = torch.zeros(tokens, heads, RANK + 32, dtype=torch.bfloat16, device="cuda")
    out = backing[..., :RANK]
    returned = mxfp4_sparse_mla_attention(q, cache, topk_slots, None, out=out, **kwargs)
    assert returned is out
    assert torch.equal(out, expected)
    assert not backing[..., RANK:].any()
