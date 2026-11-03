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

"""The gfx950 Gluon GQA prologue writes the Triton prologue's bytes, is the
gfx950 selection where its traits cover the request, and declines the rest."""

import pytest
import torch
from attention_prologue_helpers import (
    BF16,
    bytes_equal,
    cos_sin_cache,
    head_norm,
    qkv,
    run_gqa,
    slots,
)
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("The Gluon GQA prologue targets gfx950", allow_module_level=True)

from tokenspeed_kernel.ops.attention.prologue import (  # noqa: E402
    FP8TensorScales,
    HeadKVCache,
    KVCacheFormat,
    RopeStyle,
    Rotary,
    gqa_prologue,
)
from tokenspeed_kernel.selection import NoKernelFoundError  # noqa: E402


def _rotary(rope: bool, tokens: int, head_dim: int, seed: int) -> Rotary | None:
    if not rope:
        return None
    g = torch.Generator(device="cuda").manual_seed(seed)
    positions = torch.randint(0, 4096, (tokens,), device="cuda", generator=g)
    return Rotary(cos_sin_cache(head_dim), positions, RopeStyle.GPTJ, None)


# Tokens: one row, a Muse verify block, two blocks (two token tiles of the
# Gluon kernel) and a ragged prefill-sized count; heads: Muse Spark's rank
# (16 q / 2 KV of 64) and a 128-wide head.
@pytest.mark.parametrize("tokens", [1, 8, 16, 37])
@pytest.mark.parametrize("hq,hkv,head_dim", [(16, 2, 64), (8, 1, 128)])
@pytest.mark.parametrize("rope", [True, False], ids=["gptj", "nope"])
@pytest.mark.parametrize("fmt", [KVCacheFormat.NATIVE, KVCacheFormat.FP8])
@pytest.mark.parametrize("weight_offset", [0.0, 1.0])
@pytest.mark.parametrize("strided_q", [False, True])
def test_gluon_writes_the_triton_bytes(
    tokens, hq, hkv, head_dim, rope, fmt, weight_offset, strided_q
):
    total = 64
    inputs = qkv(tokens, hq, hkv, head_dim, seed=tokens + head_dim)
    inputs[0, 5] = float("nan")
    inputs[-1, hq * head_dim + 3] = -0.0
    common = dict(
        norm=head_norm(head_dim, weight_offset, seed=3),
        rotary=_rotary(rope, tokens, head_dim, seed=4),
        fmt=fmt,
        return_kv=False,
        slots=slots(tokens, total, seed=5),
        total=total,
        strided_q=strided_q,
    )
    triton = run_gqa("triton", inputs, hq, hkv, head_dim, **common)
    gluon = run_gqa("gluon", inputs, hq, hkv, head_dim, **common)
    for expected, actual in zip(triton, gluon):
        assert bytes_equal(actual, expected)


@pytest.mark.parametrize("query_dtype", [torch.float8_e4m3fn, BF16])
def test_gluon_scaled_fp8_cache_and_query_match_triton(query_dtype):
    """The Muse Spark serving case: an FP8 query and a per-tensor scaled FP8
    cache."""
    tokens, hq, hkv, dim, total = 8, 16, 2, 64, 32
    q, k, v = qkv(tokens, hq, hkv, dim, seed=9).split(
        [hq * dim, hkv * dim, hkv * dim], dim=-1
    )
    norm = head_norm(dim, 0.0, seed=10)
    rotary = _rotary(True, tokens, dim, seed=11)
    out_slots = slots(tokens, total, seed=12)
    results = []
    for solution in ("triton", "gluon"):
        k_cache = torch.full((total, hkv, dim), 0x7F, dtype=torch.uint8, device="cuda")
        v_cache = k_cache.clone()
        k_cache, v_cache = (c.view(torch.float8_e4m3fn) for c in (k_cache, v_cache))
        out = gqa_prologue(
            q, k, v, norm=norm, rotary=rotary,
            cache=HeadKVCache(k_cache, v_cache, FP8TensorScales(0.75, 1.5), out_slots),
            query_dtype=query_dtype, return_kv=False, solution=solution, override=None,
        )  # fmt: skip
        results.append((out.q, k_cache, v_cache))
    for expected, actual in zip(*results):
        assert bytes_equal(actual, expected)


def test_gluon_is_the_gfx950_selection():
    from tokenspeed_kernel.ops.attention.prologue import _select

    traits = {
        "head_dim": 64,
        "token_heads": 8 * 16,
        "has_norm": True,
        "kv_format": "fp8_per_tensor",
        "kv_convert": False,
        "mrope": False,
        "partial_rotary": False,
        "return_kv": False,
        "rope_style": "gptj",
    }
    kernel = _select("gqa_prologue", BF16, tuple(traits.items()), None, None)
    assert kernel.name == "gluon_gqa_prologue_gfx950"


@pytest.mark.parametrize(
    "change",
    [
        {"has_norm": False},
        {"rope_style": "neox"},
        {"partial_rotary": True},
        {"mrope": True},
        {"return_kv": True},
        {"kv_format": "mxfp8"},
        {"head_dim": 256},
    ],
)
def test_gluon_declines_what_it_does_not_implement(change):
    from tokenspeed_kernel.ops.attention.prologue import _select

    traits = {
        "head_dim": 64,
        "token_heads": 8 * 16,
        "has_norm": True,
        "kv_format": "fp8",
        "kv_convert": False,
        "mrope": False,
        "partial_rotary": False,
        "return_kv": False,
        "rope_style": "gptj",
    }
    traits.update(change)
    with pytest.raises((NoKernelFoundError, ValueError)):
        _select("gqa_prologue", BF16, tuple(traits.items()), "gluon", None)


@pytest.mark.parametrize("rows", [8, 16])
def test_head_tasks_on_summed_partials_write_the_triton_bytes(rows):
    """The fused layer's HEAD tasks sum the QKV task's 16 K-part partials in
    part order, round once to BF16 and run the prologue in registers: the
    bytes equal the Triton prologue's on those BF16 rows."""
    from native_body_helpers import head_task_kernel
    from tokenspeed_kernel_amd.ops.gfx950.persistent import (
        ERROR_WORDS,
        persistent_compile_options,
    )

    hq, hkv, dim, total, tag = 16, 2, 64, 64, 11
    n = (hq + 2 * hkv) * dim + hq * dim  # q | k | v | gate
    g = torch.Generator(device="cuda").manual_seed(rows)
    partials = torch.randn(16, rows, n, device="cuda", generator=g)
    words = (torch.full_like(partials, 0).to(torch.int64) + tag) << 32
    words |= partials.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    summed = partials[0].clone()
    for part in range(1, 16):
        summed += partials[part]
    qkvg = summed.to(BF16)
    norm = head_norm(dim, 0.0, seed=20)
    rotary = _rotary(True, rows, dim, seed=21)
    out_slots = slots(rows, total, seed=22)

    def cache():
        return tuple(
            torch.full((total, hkv, dim), 0x7F, dtype=torch.uint8, device="cuda").view(
                torch.float8_e4m3fn
            )
            for _ in range(2)
        )

    k_ref, v_ref = cache()
    expected = gqa_prologue(
        qkvg[:, : hq * dim], qkvg[:, hq * dim : (hq + hkv) * dim],
        qkvg[:, (hq + hkv) * dim : (hq + 2 * hkv) * dim], norm=norm, rotary=rotary,
        cache=HeadKVCache(k_ref, v_ref, None, out_slots),
        query_dtype=torch.float8_e4m3fn, return_kv=False, solution="triton",
        override=None,
    )  # fmt: skip
    k_cache, v_cache = cache()
    query = torch.empty(rows, hq * dim, dtype=torch.float8_e4m3fn, device="cuda")
    errors = torch.zeros(ERROR_WORDS, dtype=torch.int32, device="cuda")
    head_task_kernel[(hq + hkv,)](
        words.reshape(-1), query, norm.q_weight, norm.k_weight,
        rotary.cos_sin_cache, rotary.positions, k_cache, v_cache, out_slots,
        errors, tag, 2_000_000_000, rows, NUM_Q_HEADS=hq, NUM_KV_HEADS=hkv,
        HEAD_DIM=dim, num_warps=8, **persistent_compile_options(),
    )  # fmt: skip
    torch.cuda.synchronize()
    assert int(errors.abs().sum()) == 0
    assert bytes_equal(query, expected.q)
    assert bytes_equal(k_cache, k_ref) and bytes_equal(v_cache, v_ref)
