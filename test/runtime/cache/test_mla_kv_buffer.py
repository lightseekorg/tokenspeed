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

import os
import sys

import pytest
import torch
from tokenspeed_kernel.ops.embedding import (
    FusedMLASetKVBufferArg,
    apply_rope,
    apply_rope_mla,
    apply_rope_mla_set_kv,
)
from tokenspeed_kernel.ops.kvcache.triton import (
    get_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton,
)

from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
from tokenspeed.runtime.models.utils import (
    create_fused_mla_set_kv_buffer_arg,
)

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=15, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# K2.5 / DSv3 MLA dims.
NOPE_DIM = 512
ROPE_DIM = 64
TOTAL_DIM = NOPE_DIM + ROPE_DIM
NUM_PAGES = 50_000

# Spans both dispatch branches (small n -> block-split, large n -> per-loc).
N_LOC_SMALL = [4, 64, 128, 256, 511]
N_LOC_LARGE = [512, 1024, 4096]
N_LOC_ALL = N_LOC_SMALL + N_LOC_LARGE


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _make_inputs(n_loc: int, dtype: torch.dtype, pattern: str, seed: int = 0):
    torch.manual_seed(seed)
    device = "cuda"
    if pattern == "seq":
        loc = torch.arange(n_loc, device=device, dtype=torch.int64)
    else:
        loc = torch.randperm(NUM_PAGES, device=device, dtype=torch.int64)[:n_loc]

    if dtype == torch.float8_e4m3fn:
        bf = torch.randn(n_loc, 1, NOPE_DIM, device=device, dtype=torch.bfloat16) * 50
        k_nope = bf.to(dtype)
        bf = torch.randn(n_loc, 1, ROPE_DIM, device=device, dtype=torch.bfloat16) * 50
        k_rope = bf.to(dtype)
    else:
        k_nope = torch.randn(n_loc, 1, NOPE_DIM, device=device, dtype=dtype)
        k_rope = torch.randn(n_loc, 1, ROPE_DIM, device=device, dtype=dtype)
    return loc, k_nope, k_rope


def _empty_kv(dtype: torch.dtype) -> torch.Tensor:
    """Allocate an unused-cell sentinel-filled kv_buffer so untouched cells
    diverge if the kernel writes to them."""
    sentinel = torch.full(
        (NUM_PAGES, TOTAL_DIM), 7.5, device="cuda", dtype=torch.bfloat16
    )
    return sentinel.to(dtype) if dtype == torch.float8_e4m3fn else sentinel.to(dtype)


def _torch_set_reference(kv: torch.Tensor, loc, k_nope, k_rope) -> torch.Tensor:
    """Pure-torch scatter-write reference."""
    out = kv.clone()
    out[loc, :NOPE_DIM] = k_nope[:, 0, :]
    out[loc, NOPE_DIM:] = k_rope[:, 0, :]
    return out


def _torch_get_reference(kv: torch.Tensor, loc) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch scatter-read reference."""
    return (
        kv[loc, :NOPE_DIM].unsqueeze(1).contiguous(),
        kv[loc, NOPE_DIM:].unsqueeze(1).contiguous(),
    )


def _rotate_rope_reference(
    x: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool,
) -> torch.Tensor:
    cos, sin = cos_sin[positions].chunk(2, dim=-1)
    half = x.shape[-1] // 2
    x_float = x.float()
    cos = cos.float().unsqueeze(-2)
    sin = sin.float().unsqueeze(-2)
    if is_neox:
        x1 = x_float[..., :half]
        x2 = x_float[..., half:]
        out = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    else:
        x1 = x_float[..., 0::2]
        x2 = x_float[..., 1::2]
        out = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        out = out.flatten(-2)
    return out.to(x.dtype)


# ─── set ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_loc", N_LOC_ALL)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("pattern", ["seq", "rand"])
def test_set_matches_torch_reference(n_loc, dtype, pattern):
    """set_mla_kv_buffer_triton scatters k_nope/k_rope into kv_buffer at loc
    indices, byte-for-byte vs a torch reference. Spans both dispatch branches
    via the n_loc parametrization."""
    loc, k_nope, k_rope = _make_inputs(n_loc, dtype, pattern)
    kv = _empty_kv(dtype)
    ref = _torch_set_reference(kv, loc, k_nope, k_rope)

    set_mla_kv_buffer_triton(kv, loc, k_nope, k_rope)
    torch.cuda.synchronize()

    assert _bitwise_equal(kv, ref)


@pytest.mark.parametrize("n_loc", [4, 600])
@pytest.mark.parametrize("pattern", ["seq", "rand"])
def test_set_casts_bf16_sources_into_fp8_buffer(n_loc, pattern):
    """The write kernel casts to the buffer dtype on store, so bf16 sources
    land in an fp8 buffer byte-for-byte as if pre-cast with torch. Spans both
    dispatch branches via the n_loc parametrization."""
    loc, k_nope, k_rope = _make_inputs(n_loc, torch.bfloat16, pattern)
    kv = _empty_kv(torch.float8_e4m3fn)
    ref = _torch_set_reference(
        kv, loc, k_nope.to(torch.float8_e4m3fn), k_rope.to(torch.float8_e4m3fn)
    )

    set_mla_kv_buffer_triton(kv, loc, k_nope, k_rope)
    torch.cuda.synchronize()

    assert _bitwise_equal(kv, ref)


@pytest.mark.parametrize("n_loc", [4, 600])
def test_set_squashes_nan_and_inf(n_loc):
    """Sanitized mixed-dtype writes stay finite in the fp8 destination."""
    loc, k_nope, k_rope = _make_inputs(n_loc, torch.bfloat16, "rand")
    k_nope[0, 0, 0] = float("nan")
    k_nope[0, 0, 1] = float("inf")
    k_rope[0, 0, 0] = float("-inf")
    kv = _empty_kv(torch.float8_e4m3fn)
    ref = _torch_set_reference(
        kv,
        loc,
        torch.nan_to_num(k_nope.float(), nan=0.0, posinf=448.0, neginf=-448.0).to(
            torch.float8_e4m3fn
        ),
        torch.nan_to_num(k_rope.float(), nan=0.0, posinf=448.0, neginf=-448.0).to(
            torch.float8_e4m3fn
        ),
    )

    set_mla_kv_buffer_triton(kv, loc, k_nope, k_rope, sanitize=True)
    torch.cuda.synchronize()

    assert _bitwise_equal(kv, ref)
    assert not torch.isnan(kv[loc.cpu()].float()).any()


@pytest.mark.parametrize("n_loc", [4, 511, 512, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_set_pdl_invariant(n_loc, dtype):
    """PDL is a scheduling hint; output must be bitwise-identical regardless."""
    loc, k_nope, k_rope = _make_inputs(n_loc, dtype, "rand")
    kv_off = _empty_kv(dtype)
    kv_on = _empty_kv(dtype)

    set_mla_kv_buffer_triton(kv_off, loc, k_nope, k_rope, enable_pdl=False)
    set_mla_kv_buffer_triton(kv_on, loc, k_nope, k_rope, enable_pdl=True)
    torch.cuda.synchronize()

    assert _bitwise_equal(kv_off, kv_on)


# ─── get ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_loc", N_LOC_ALL)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("pattern", ["seq", "rand"])
def test_get_matches_torch_reference(n_loc, dtype, pattern):
    """get_mla_kv_buffer_triton gathers from kv_buffer at loc indices into
    cache_k_nope / cache_k_rope outputs, byte-for-byte vs a torch reference."""
    # Populate kv_buffer with random data we'll read back.
    if dtype == torch.float8_e4m3fn:
        bf = torch.randn(NUM_PAGES, TOTAL_DIM, device="cuda", dtype=torch.bfloat16) * 50
        kv = bf.to(dtype)
    else:
        kv = torch.randn(NUM_PAGES, TOTAL_DIM, device="cuda", dtype=dtype)

    if pattern == "seq":
        loc = torch.arange(n_loc, device="cuda", dtype=torch.int64)
    else:
        loc = torch.randperm(NUM_PAGES, device="cuda", dtype=torch.int64)[:n_loc]

    k_nope = torch.empty((n_loc, 1, NOPE_DIM), dtype=dtype, device="cuda")
    k_rope = torch.empty((n_loc, 1, ROPE_DIM), dtype=dtype, device="cuda")
    nope_ref, rope_ref = _torch_get_reference(kv, loc)

    get_mla_kv_buffer_triton(kv, loc, k_nope, k_rope)
    torch.cuda.synchronize()

    assert _bitwise_equal(k_nope, nope_ref)
    assert _bitwise_equal(k_rope, rope_ref)


@pytest.mark.parametrize("n_loc", [4, 511, 512, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_get_pdl_invariant(n_loc, dtype):
    if dtype == torch.float8_e4m3fn:
        bf = torch.randn(NUM_PAGES, TOTAL_DIM, device="cuda", dtype=torch.bfloat16) * 50
        kv = bf.to(dtype)
    else:
        kv = torch.randn(NUM_PAGES, TOTAL_DIM, device="cuda", dtype=dtype)

    loc = torch.randperm(NUM_PAGES, device="cuda", dtype=torch.int64)[:n_loc]

    k_nope_off = torch.empty((n_loc, 1, NOPE_DIM), dtype=dtype, device="cuda")
    k_rope_off = torch.empty((n_loc, 1, ROPE_DIM), dtype=dtype, device="cuda")
    k_nope_on = torch.empty_like(k_nope_off)
    k_rope_on = torch.empty_like(k_rope_off)

    get_mla_kv_buffer_triton(kv, loc, k_nope_off, k_rope_off, enable_pdl=False)
    get_mla_kv_buffer_triton(kv, loc, k_nope_on, k_rope_on, enable_pdl=True)
    torch.cuda.synchronize()

    assert _bitwise_equal(k_nope_off, k_nope_on)
    assert _bitwise_equal(k_rope_off, k_rope_on)


# ─── round trip ─────────────────────────────────────────────────────


@pytest.mark.parametrize("n_loc", [128, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_set_then_get_round_trip(n_loc, dtype):
    """set followed by get on the same loc indices recovers the original
    k_nope / k_rope inputs bitwise."""
    loc, k_nope_in, k_rope_in = _make_inputs(n_loc, dtype, "rand")
    kv = _empty_kv(dtype)

    set_mla_kv_buffer_triton(kv, loc, k_nope_in, k_rope_in)

    k_nope_out = torch.empty_like(k_nope_in)
    k_rope_out = torch.empty_like(k_rope_in)
    get_mla_kv_buffer_triton(kv, loc, k_nope_out, k_rope_out)
    torch.cuda.synchronize()

    assert _bitwise_equal(k_nope_out, k_nope_in)
    assert _bitwise_equal(k_rope_out, k_rope_in)


@pytest.mark.parametrize("is_neox", [False, True])
@pytest.mark.parametrize("loc_dtype", [torch.int32, torch.int64])
def test_mla_rope_set_kv_buffer_matches_reference(is_neox, loc_dtype):
    torch.manual_seed(0)
    n_loc = 17
    num_heads = 3
    max_position = 128
    device = "cuda"
    dtype = torch.bfloat16

    q_rope = torch.randn(n_loc, num_heads, ROPE_DIM, device=device, dtype=dtype)
    k_nope = torch.randn(n_loc, 1, NOPE_DIM, device=device, dtype=dtype)
    k_rope = torch.randn(n_loc, 1, ROPE_DIM, device=device, dtype=dtype)
    q_out_rope = torch.empty_like(q_rope)
    kv = _empty_kv(dtype)
    loc = torch.randperm(NUM_PAGES, device=device, dtype=loc_dtype)[:n_loc]
    positions = torch.randint(
        0, max_position, (n_loc,), device=device, dtype=torch.int64
    )

    angles = torch.randn(max_position, ROPE_DIM, device=device, dtype=torch.float32)
    cos_sin = torch.cat(
        (torch.cos(angles[:, : ROPE_DIM // 2]), torch.sin(angles[:, : ROPE_DIM // 2])),
        dim=-1,
    )

    q_ref = _rotate_rope_reference(q_rope, cos_sin, positions, is_neox)
    k_rope_ref = _rotate_rope_reference(k_rope, cos_sin, positions, is_neox)
    kv_ref = kv.clone()
    kv_ref[loc.long(), :NOPE_DIM] = k_nope[:, 0, :]
    kv_ref[loc.long(), NOPE_DIM:] = k_rope_ref[:, 0, :]

    apply_rope(
        cos_sin_cache=cos_sin,
        fused_mla_set_kv_buffer_arg=FusedMLASetKVBufferArg(
            k_nope=k_nope,
            kv_buffer=kv,
            cache_loc=loc,
        ),
        head_size=ROPE_DIM,
        positions=positions,
        q=q_rope,
        k=k_rope,
        q_rope_out=q_out_rope,
        is_neox=is_neox,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(q_out_rope, q_ref, atol=0.01, rtol=0.01)
    torch.testing.assert_close(kv[loc.long()], kv_ref[loc.long()], atol=0.01, rtol=0.01)


def test_mla_rope_set_kv_buffer_fp8_matches_two_kernel_path() -> None:
    torch.manual_seed(0)
    n_loc = 17
    num_heads = 3
    max_position = 128
    dtype = torch.bfloat16

    q_rope = torch.randn(n_loc, num_heads, ROPE_DIM, device="cuda", dtype=dtype)
    k_rope = torch.randn(n_loc, 1, ROPE_DIM, device="cuda", dtype=dtype)
    q_nope = torch.randn(n_loc, num_heads, NOPE_DIM, device="cuda", dtype=dtype)
    k_nope = torch.randn(n_loc, 1, NOPE_DIM, device="cuda", dtype=dtype)
    positions = torch.randint(0, max_position, (n_loc,), device="cuda")
    loc = torch.randperm(NUM_PAGES, device="cuda")[:n_loc]
    angles = torch.randn(max_position, ROPE_DIM // 2, device="cuda")
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)

    query_ref, key_ref = apply_rope_mla(
        positions=positions,
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=cos_sin,
    )
    query = torch.empty_like(query_ref)
    kv = _empty_kv(torch.float8_e4m3fn)

    apply_rope(
        positions=positions,
        q=q_rope,
        k=k_rope,
        head_size=ROPE_DIM,
        cos_sin_cache=cos_sin,
        fused_mla_set_kv_buffer_arg=FusedMLASetKVBufferArg(
            k_nope=k_nope,
            kv_buffer=kv,
            cache_loc=loc,
            q_nope=q_nope,
        ),
        q_rope_out=query,
    )
    torch.cuda.synchronize()

    assert _bitwise_equal(query, query_ref)
    assert _bitwise_equal(kv[loc], key_ref[:, 0])


@pytest.mark.parametrize("n_loc", [1, 17, 600])
def test_mla_set_kv_nope_matches_two_kernel_path(n_loc: int) -> None:
    """The NoPE form, through the entry point the model actually calls.

    A model with no rotary embedding hands the write no RoPE tables, so the
    halves are assembled without rotation. n_loc spans both sides of the point
    where a program starts covering several tokens.
    """
    torch.manual_seed(0)
    num_heads = 3
    dtype = torch.bfloat16

    q_rope = torch.randn(n_loc, num_heads, ROPE_DIM, device="cuda", dtype=dtype)
    k_rope = torch.randn(n_loc, 1, ROPE_DIM, device="cuda", dtype=dtype)
    q_nope = torch.randn(n_loc, num_heads, NOPE_DIM, device="cuda", dtype=dtype)
    k_nope = torch.randn(n_loc, 1, NOPE_DIM, device="cuda", dtype=dtype)
    positions = torch.zeros(n_loc, device="cuda", dtype=torch.int64)
    loc = torch.randperm(NUM_PAGES, device="cuda")[:n_loc]

    query_ref, key_ref = apply_rope_mla(
        positions=positions,
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=None,
        is_neox=False,
    )
    query = torch.empty_like(query_ref)
    kv = _empty_kv(torch.float8_e4m3fn)

    apply_rope_mla_set_kv(
        positions=positions,
        q_rope=q_rope,
        k_rope=k_rope,
        fused_mla_set_kv_buffer_arg=FusedMLASetKVBufferArg(
            k_nope=k_nope,
            kv_buffer=kv,
            cache_loc=loc,
            q_nope=q_nope,
            cos_sin_cache=None,
        ),
        q_rope_out=query,
    )
    torch.cuda.synchronize()

    assert _bitwise_equal(query, query_ref)
    assert _bitwise_equal(kv[loc], key_ref[:, 0])


@pytest.mark.parametrize("n_loc", [4, 600])
@pytest.mark.parametrize("has_rope", [False, True])
def test_fused_sanitize_matches_the_split_path_bytes(
    n_loc: int, has_rope: bool
) -> None:
    """A sanitizing fused write lands the same latent bytes as the split path.

    This is the property that lets Kimi-K3's pool declare
    ``latent_write_sanitizes`` instead of overriding the latent write: the
    hazard the override closed (a padded row's NaN reaching a live row's
    softmax through the shared dummy slot) stays closed on the fused path.
    Both RoPE forms are covered because the rotation mixes the two halves, so
    where the clamp lands relative to it is observable.
    """
    torch.manual_seed(0)
    num_heads = 3
    dtype = torch.bfloat16

    q_rope = torch.randn(n_loc, num_heads, ROPE_DIM, device="cuda", dtype=dtype)
    k_rope = torch.randn(n_loc, 1, ROPE_DIM, device="cuda", dtype=dtype)
    q_nope = torch.randn(n_loc, num_heads, NOPE_DIM, device="cuda", dtype=dtype)
    k_nope = torch.randn(n_loc, 1, NOPE_DIM, device="cuda", dtype=dtype)
    # Every non-finite the padded rows can carry, in both halves.
    k_nope[0, 0, 0] = float("nan")
    k_nope[0, 0, 1] = float("inf")
    k_nope[0, 0, 2] = float("-inf")
    k_rope[0, 0, 0] = float("nan")
    k_rope[0, 0, 1] = float("inf")
    k_rope[0, 0, ROPE_DIM // 2] = float("-inf")

    positions = torch.zeros(n_loc, device="cuda", dtype=torch.int64)
    loc = torch.randperm(NUM_PAGES, device="cuda")[:n_loc]
    cos_sin_cache = (
        torch.randn(2048, ROPE_DIM, device="cuda", dtype=torch.float32)
        if has_rope
        else None
    )

    # Split path: rope+quantize, then the sanitizing scatter.
    _, key_ref = apply_rope_mla(
        positions=positions,
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=cos_sin_cache,
        is_neox=False,
    )
    kv_ref = _empty_kv(torch.float8_e4m3fn)
    set_mla_kv_buffer_triton(
        kv_ref,
        loc,
        key_ref[:, :, :NOPE_DIM],
        key_ref[:, :, NOPE_DIM:],
        sanitize=True,
    )

    kv = _empty_kv(torch.float8_e4m3fn)
    query = torch.empty(
        n_loc, num_heads, NOPE_DIM + ROPE_DIM, device="cuda", dtype=dtype
    )
    apply_rope_mla_set_kv(
        positions=positions,
        q_rope=q_rope,
        k_rope=k_rope,
        fused_mla_set_kv_buffer_arg=FusedMLASetKVBufferArg(
            k_nope=k_nope,
            kv_buffer=kv,
            cache_loc=loc,
            q_nope=q_nope,
            cos_sin_cache=cos_sin_cache,
            is_neox=False,
            sanitize=True,
        ),
        q_rope_out=query,
    )
    torch.cuda.synchronize()

    assert not torch.isnan(kv[loc.cpu()].float()).any()
    assert torch.isfinite(kv[loc.cpu()].float()).all()
    assert _bitwise_equal(kv, kv_ref)


def _fake_mla_pool(dtype: torch.dtype = torch.float8_e4m3fn) -> MLATokenToKVPool:
    """A minimal MLATokenToKVPool that isinstance-checks true, for gate tests
    that need no real cache -- construction takes eight unrelated args."""
    pool = object.__new__(MLATokenToKVPool)
    pool.quant_method = "none"
    pool.dtype = dtype
    pool.store_dtype = dtype
    pool.layerwise_load_tracker = None
    pool.kv_buffer = [torch.zeros(NUM_PAGES, 1, TOTAL_DIM, dtype=dtype, device="cuda")]
    return pool


@pytest.mark.parametrize(
    "num_q_heads,num_tokens,expect_fused",
    [
        (16, 2048, True),
        (16, 2049, False),
        (32, 1024, True),
        (32, 1025, False),
        (64, 512, True),
        (64, 513, False),
    ],
)
def test_fused_gate_token_head_budget(num_q_heads, num_tokens, expect_fused):
    """The fused write's token cap scales down with head count: the fused
    kernel's own CTA count is tokens*(heads+1), so a fixed token cap is only
    safe at the head count it was measured at. Regression measured directly:
    at H=32 the fused path lost to the split path by 18.9% at 2048 tokens
    while still winning at 1024; at H=64 it lost by 4.9% at 768 while still
    winning at 512."""
    pool = _fake_mla_pool()
    k_nope = torch.zeros(num_tokens, 1, NOPE_DIM, device="cuda", dtype=torch.bfloat16)
    q_nope = torch.zeros(
        num_tokens, num_q_heads, NOPE_DIM, device="cuda", dtype=torch.bfloat16
    )
    loc = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    arg = create_fused_mla_set_kv_buffer_arg(
        k_nope=k_nope,
        rope_dim=ROPE_DIM,
        rotary_emb=None,
        out_cache_loc=loc,
        token_to_kv_pool=pool,
        layer_id=0,
        num_q_heads=num_q_heads,
        q_nope=q_nope,
    )
    assert (arg is not None) == expect_fused


def _gate_arg_for_pool(pool, n_loc: int = 8, num_q_heads: int = 16):
    """Run the fused-write gate against ``pool`` with an otherwise valid shape."""
    return create_fused_mla_set_kv_buffer_arg(
        k_nope=torch.zeros(n_loc, 1, NOPE_DIM, device="cuda", dtype=torch.bfloat16),
        rope_dim=ROPE_DIM,
        rotary_emb=None,
        out_cache_loc=torch.arange(n_loc, device="cuda", dtype=torch.int64),
        token_to_kv_pool=pool,
        layer_id=0,
        num_q_heads=num_q_heads,
        q_nope=torch.zeros(
            n_loc, num_q_heads, NOPE_DIM, device="cuda", dtype=torch.bfloat16
        ),
    )


def test_fused_gate_declines_a_pool_that_overrides_the_latent_write():
    """A pool that customizes set_mla_kv_buffer must not be fused past.

    An override adds something the fused write does not have, and fusing would
    bypass it silently. isinstance alone does not exclude such a pool, since it
    is an MLATokenToKVPool subclass -- the gate compares method identity.
    """

    class _OverridingPool(MLATokenToKVPool):
        def set_mla_kv_buffer(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("must not be reached")

    pool = _fake_mla_pool()
    pool.__class__ = _OverridingPool

    assert issubclass(_OverridingPool, MLATokenToKVPool)
    assert _gate_arg_for_pool(pool) is None


def test_fused_gate_admits_the_hybrid_kda_pool_and_carries_its_sanitize():
    """Kimi-K3's pool reaches the fused write, sanitizing.

    It needs a sanitizing latent write -- a padded row's NaN otherwise reaches
    a live row's softmax through the shared dummy slot -- but declares that
    through ``latent_write_sanitizes`` instead of overriding the method, so the
    identity check above still holds and the flag rides into the kernel.
    """
    from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
        HybridKDATokenToKVPool,
    )

    assert issubclass(HybridKDATokenToKVPool, MLATokenToKVPool)
    assert (
        HybridKDATokenToKVPool.set_mla_kv_buffer is MLATokenToKVPool.set_mla_kv_buffer
    )
    assert HybridKDATokenToKVPool.latent_write_sanitizes is True
    assert MLATokenToKVPool.latent_write_sanitizes is False

    pool = _fake_mla_pool()
    pool.__class__ = HybridKDATokenToKVPool
    arg = _gate_arg_for_pool(pool)
    assert arg is not None
    assert arg.sanitize is True

    plain = _fake_mla_pool()
    plain_arg = _gate_arg_for_pool(plain)
    assert plain_arg is not None
    assert plain_arg.sanitize is False


def test_every_fused_call_site_uses_the_probed_entry_point():
    """The gate answers for ``embedding.rope_mla_set_kv``, so every launch
    that carries its argument goes through that entry point.

    Overrides are keyed by ``(family, mode)`` and resolve without trait
    filtering, so a call site that builds the argument here and then launches
    through ``embedding.rope`` -- as ``rotary_emb`` does -- can be redirected
    to a solution that rejects the fused argument, and inference raises where
    the gate promised a working kernel.
    """
    import ast
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parents[3]
        / "python/tokenspeed/runtime/models/deepseek_v3.py"
    ).read_text()
    tree = ast.parse(source)

    carriers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and any(kw.arg == "fused_mla_set_kv_buffer_arg" for kw in node.keywords)
    ]
    assert len(carriers) == 2, "expected the FP8 and BF16 fused branches"
    for call in carriers:
        assert isinstance(call.func, ast.Name), ast.dump(call.func)
        assert call.func.id == "apply_rope_mla_set_kv", call.func.id


@pytest.mark.parametrize("n_loc", [1, 4, 600])
def test_fused_write_follows_a_strided_cache_loc(n_loc: int) -> None:
    """A strided ``cache_loc`` lands the same rows as a packed one.

    A column of a per-step location table arrives with a stride of
    ``num_draft_tokens``; indexing it as ``ptr + token_idx`` would read a
    different row per token and write the latent to the wrong pages. n_loc=1
    is the shape that matters most: a one-element column reports that stride
    while ``is_contiguous()`` is True, so it is legitimate input that a
    stride-1 assertion rejects.
    """
    torch.manual_seed(0)
    num_heads = 3
    num_draft_tokens = 4
    dtype = torch.bfloat16

    q_rope = torch.randn(n_loc, num_heads, ROPE_DIM, device="cuda", dtype=dtype)
    k_rope = torch.randn(n_loc, 1, ROPE_DIM, device="cuda", dtype=dtype)
    q_nope = torch.randn(n_loc, num_heads, NOPE_DIM, device="cuda", dtype=dtype)
    k_nope = torch.randn(n_loc, 1, NOPE_DIM, device="cuda", dtype=dtype)
    positions = torch.zeros(n_loc, device="cuda", dtype=torch.int64)

    table = torch.randperm(NUM_PAGES, device="cuda")[: n_loc * num_draft_tokens]
    strided_loc = table.view(n_loc, num_draft_tokens)[:, 1]
    assert strided_loc.stride(-1) == num_draft_tokens
    packed_loc = strided_loc.contiguous()

    def run(loc):
        query = torch.empty(
            n_loc, num_heads, NOPE_DIM + ROPE_DIM, device="cuda", dtype=dtype
        )
        kv = _empty_kv(torch.float8_e4m3fn)
        apply_rope_mla_set_kv(
            positions=positions,
            q_rope=q_rope,
            k_rope=k_rope,
            fused_mla_set_kv_buffer_arg=FusedMLASetKVBufferArg(
                k_nope=k_nope,
                kv_buffer=kv,
                cache_loc=loc,
                q_nope=q_nope,
                cos_sin_cache=None,
            ),
            q_rope_out=query,
        )
        torch.cuda.synchronize()
        return query, kv

    query_strided, kv_strided = run(strided_loc)
    query_packed, kv_packed = run(packed_loc)

    assert _bitwise_equal(query_strided, query_packed)
    assert _bitwise_equal(kv_strided[packed_loc], kv_packed[packed_loc])
