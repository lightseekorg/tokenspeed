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

"""FP8 softmax probabilities in the gfx950 MHA decode, extend and prefill kernels.

The first key has a base-2 logit of 12. The others either score 0, each
with 2^-12 of the first key's probability, 4096 of them, or score 6, each
with 2^-6, 64 of them (short ranges). Either way they hold as much softmax
mass as the first key, and e4m3 rounds 2^-12 to zero; the kernels must keep
it. The first key's value is 0 and the others' 1, so every output element is
the others' share of the mass: about 1/2, or 1/3 with a sink of the first
key's logit, which is wrong by a factor of 256 if a sink misses the FP8
probability scale. All inputs are exact in e4m3.
"""

from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.mha import torch_dense_attention
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx950.attention.mha.decode import (  # noqa: E402
    launch_gluon_mha_decode_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.mha.extend import (  # noqa: E402
    launch_gluon_mha_extend_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.mha.prefill import (  # noqa: E402
    launch_gluon_mha_prefill_gfx950,
)

FP8 = torch.float8_e4m3fn
# The head dims extend and prefill accept.
HEAD_DIMS = [64, 128]
Q_HEADS = 16
KV_HEADS = 2
PAGE_SIZE = 64
FIRST_LOGIT = 12.0
OTHER_KEYS = 4096
SHORT_OTHER_KEYS = 64
SHORT_OTHER_LOGIT = 6.0
# Logit scale ln 2, so a dot product is the logit in base 2.
SOFTMAX_SCALE = math.log(2.0)


def _keys_and_values(
    num_keys: int, other_logit: float, head_dim: int
) -> tuple[torch.Tensor, ...]:
    """[num_keys, KV_HEADS, head_dim] K and V: key 0 scores 12, the rest other_logit."""
    k = torch.zeros(num_keys, KV_HEADS, head_dim, device="cuda")
    k[:, :, 0] = other_logit
    k[0, :, 0] = FIRST_LOGIT
    v = torch.ones(num_keys, KV_HEADS, head_dim, device="cuda")
    v[0] = 0.0
    return k.to(FP8), v.to(FP8)


def _queries(rows: int, head_dim: int) -> torch.Tensor:
    q = torch.zeros(rows, Q_HEADS, head_dim, device="cuda")
    q[:, :, 0] = 1.0
    return q.to(FP8)


def _sinks(has_sink: bool) -> torch.Tensor | None:
    """A per-head sink logit (natural log) equal to the first key's."""
    if not has_sink:
        return None
    return torch.full((Q_HEADS,), FIRST_LOGIT * math.log(2.0), device="cuda")


def _paged(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """One request's keys in pages, in reverse page order."""
    pages = -(-k.shape[0] // PAGE_SIZE)
    pad = pages * PAGE_SIZE - k.shape[0]

    def to_pages(x):
        x = torch.cat([x, x.new_zeros(pad, *x.shape[1:])]).view(
            pages, PAGE_SIZE, KV_HEADS, x.shape[-1]
        )
        return x.flip(0).contiguous()

    page_table = torch.arange(pages - 1, -1, -1, dtype=torch.int32, device="cuda")
    return to_pages(k), to_pages(v), page_table[None]


def _check(out: torch.Tensor, expected: torch.Tensor, has_sink: bool) -> None:
    # The others hold about half of the mass (a third with the sink); zeroed
    # probabilities give 0, an unscaled sink about 1/2, an overscaled one 0.004.
    assert expected.min() > (0.3 if has_sink else 0.45)
    torch.testing.assert_close(out.float(), expected, atol=2e-3, rtol=0)


# (window_left, other keys, their logit, queries per request): full attention
# and a wide window split the KV stream; a window of one page does not.
DECODE_CASES = [
    pytest.param(-1, OTHER_KEYS, 0.0, 1, id="full-q1"),
    pytest.param(-1, OTHER_KEYS, 0.0, 8, id="full-q8"),
    pytest.param(OTHER_KEYS + 16, OTHER_KEYS, 0.0, 1, id="window-split-q1"),
    pytest.param(OTHER_KEYS + 16, OTHER_KEYS, 0.0, 8, id="window-split-q8"),
    pytest.param(
        SHORT_OTHER_KEYS, SHORT_OTHER_KEYS, SHORT_OTHER_LOGIT, 1, id="window-one-split"
    ),
]


@pytest.mark.parametrize("has_sink", [False, True])
@pytest.mark.parametrize("window_left,others,other_logit,max_seqlen_q", DECODE_CASES)
def test_decode_keeps_small_fp8_probabilities(
    window_left, others, other_logit, max_seqlen_q, has_sink
):
    num_keys = 1 + others
    k, v = _keys_and_values(num_keys, other_logit, 64)
    k_cache, v_cache, page_table = _paged(k, v)
    q = _queries(max_seqlen_q, 64)
    sinks = _sinks(has_sink)
    out = launch_gluon_mha_decode_gfx950(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=torch.tensor([num_keys], dtype=torch.int32, device="cuda"),
        max_seqlen_k=num_keys,
        max_seqlen_q=max_seqlen_q,
        window_left=window_left,
        sinks=sinks,
        softmax_scale=SOFTMAX_SCALE,
    )
    expected, _ = torch_dense_attention(
        q,
        k,
        v,
        scale=SOFTMAX_SCALE,
        causal_offset=num_keys - max_seqlen_q,
        window_left=None if window_left < 0 else window_left,
        sinks=sinks,
    )
    _check(out, expected, has_sink)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("has_sink", [False, True])
@pytest.mark.parametrize("window_left", [-1, OTHER_KEYS + 64])
def test_extend_keeps_small_fp8_probabilities(window_left, has_sink, head_dim):
    rows = 32
    num_keys = 1 + OTHER_KEYS + rows
    k, v = _keys_and_values(num_keys, 0.0, head_dim)
    k_cache, v_cache, page_table = _paged(k, v)
    q = _queries(rows, head_dim)
    sinks = _sinks(has_sink)
    out, lse = launch_gluon_mha_extend_gfx950(
        q=q,
        cu_seqlens_q=torch.tensor([0, rows], dtype=torch.int32, device="cuda"),
        cu_seqlens_kv=torch.tensor([0, num_keys], dtype=torch.int32, device="cuda"),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=torch.tensor([num_keys], dtype=torch.int32, device="cuda"),
        is_causal=True,
        window_left=window_left,
        sinks=sinks,
        return_lse=True,
        max_seqlen_q=rows,
        max_seqlen_k=num_keys,
        softmax_scale=SOFTMAX_SCALE,
    )
    expected, expected_lse = torch_dense_attention(
        q,
        k,
        v,
        scale=SOFTMAX_SCALE,
        causal_offset=num_keys - rows,
        window_left=None if window_left < 0 else window_left,
        sinks=sinks,
    )
    _check(out, expected, has_sink)
    if not has_sink:
        torch.testing.assert_close(lse, expected_lse, atol=1e-4, rtol=0)


# (other keys, their logit, first checked row): over 64 keys the kernel tiles
# the keys; below, one tile holds them all.
PREFILL_CASES = [
    pytest.param(OTHER_KEYS + 64, 0.0, OTHER_KEYS, id="tiled"),
    pytest.param(
        SHORT_OTHER_KEYS - 2, SHORT_OTHER_LOGIT, SHORT_OTHER_KEYS - 2, id="one-tile"
    ),
]


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("has_sink", [False, True])
@pytest.mark.parametrize("others,other_logit,first_checked_row", PREFILL_CASES)
def test_prefill_keeps_small_fp8_probabilities(
    others, other_logit, first_checked_row, has_sink, head_dim
):
    num_keys = 1 + others
    k, v = _keys_and_values(num_keys, other_logit, head_dim)
    q = _queries(num_keys, head_dim)
    sinks = _sinks(has_sink)
    out, lse = launch_gluon_mha_prefill_gfx950(
        q=q,
        k=k,
        v=v,
        cu_seqlens=torch.tensor([0, num_keys], dtype=torch.int32, device="cuda"),
        cu_seqlens_cpu=[0, num_keys],
        max_seqlen=num_keys,
        sinks=sinks,
        return_lse=True,
        softmax_scale=SOFTMAX_SCALE,
    )
    expected, expected_lse = torch_dense_attention(
        q, k, v, scale=SOFTMAX_SCALE, causal_offset=0, window_left=None, sinks=sinks
    )
    # The last rows see (nearly) all of the other keys.
    tail = slice(first_checked_row, None)
    _check(out[tail], expected[tail], has_sink)
    if not has_sink:
        torch.testing.assert_close(lse, expected_lse, atol=1e-4, rtol=0)


def _rel_l2(out: torch.Tensor, expected: torch.Tensor) -> float:
    return float((out.float() - expected).norm() / expected.norm())


# Bound on the relative L2 error of an FP8 kernel against FP32 attention over
# the same FP8 inputs: the probabilities' 3-bit FP8 rounding costs 1.0-1.4%
# on these cases (measured on MI350X).
# Random keys use every head channel, so a wrong MFMA operand layout fails it;
# the dominant-key cases above load only channel 0.
RANDOM_INPUT_REL_L2 = 2e-2


def _random_fp8(*shape: int, scale: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(*shape, device="cuda", generator=generator) * scale
    return x.to(FP8)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("window_left", [-1, 255])
@pytest.mark.parametrize("prefix,rows", [(0, 1024), (2048, 32), (2048, 1024)])
def test_extend_matches_reference_on_random_fp8(prefix, rows, window_left, head_dim):
    num_keys = prefix + rows
    k = _random_fp8(num_keys, KV_HEADS, head_dim, scale=1.0, seed=1)
    v = _random_fp8(num_keys, KV_HEADS, head_dim, scale=1.0, seed=2)
    q = _random_fp8(rows, Q_HEADS, head_dim, scale=2.0, seed=3)
    k_cache, v_cache, page_table = _paged(k, v)
    out = launch_gluon_mha_extend_gfx950(
        q=q,
        cu_seqlens_q=torch.tensor([0, rows], dtype=torch.int32, device="cuda"),
        cu_seqlens_kv=torch.tensor([0, num_keys], dtype=torch.int32, device="cuda"),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=torch.tensor([num_keys], dtype=torch.int32, device="cuda"),
        is_causal=True,
        window_left=window_left,
        max_seqlen_q=rows,
        max_seqlen_k=num_keys,
    )
    expected, _ = torch_dense_attention(
        q,
        k,
        v,
        scale=head_dim**-0.5,
        causal_offset=prefix,
        window_left=None if window_left < 0 else window_left,
        sinks=None,
    )
    assert _rel_l2(out, expected) < RANDOM_INPUT_REL_L2


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("window_left", [-1, 255])
def test_prefill_matches_reference_on_random_fp8(window_left, head_dim):
    num_keys = 2048
    k = _random_fp8(num_keys, KV_HEADS, head_dim, scale=1.0, seed=1)
    v = _random_fp8(num_keys, KV_HEADS, head_dim, scale=1.0, seed=2)
    q = _random_fp8(num_keys, Q_HEADS, head_dim, scale=2.0, seed=3)
    out = launch_gluon_mha_prefill_gfx950(
        q=q,
        k=k,
        v=v,
        cu_seqlens=torch.tensor([0, num_keys], dtype=torch.int32, device="cuda"),
        cu_seqlens_cpu=[0, num_keys],
        max_seqlen=num_keys,
        window_left=window_left,
    )
    expected, _ = torch_dense_attention(
        q,
        k,
        v,
        scale=head_dim**-0.5,
        causal_offset=0,
        window_left=None if window_left < 0 else window_left,
        sinks=None,
    )
    assert _rel_l2(out, expected) < RANDOM_INPUT_REL_L2


@pytest.mark.parametrize("scale", ["q_scale", "k_scale", "v_scale"])
def test_decode_and_extend_reject_block_scales(scale):
    k, v = _keys_and_values(PAGE_SIZE, 0.0, 64)
    k_cache, v_cache, page_table = _paged(k, v)
    q = _queries(1, 64)
    lengths = torch.tensor([PAGE_SIZE], dtype=torch.int32, device="cuda")
    block_scales = {scale: torch.ones(1, dtype=torch.uint8, device="cuda")}
    with pytest.raises(ValueError, match=f"unit scale, got {scale}"):
        launch_gluon_mha_decode_gfx950(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=lengths,
            max_seqlen_k=PAGE_SIZE,
            **block_scales,
        )
    with pytest.raises(ValueError, match=f"unit scale, got {scale}"):
        launch_gluon_mha_extend_gfx950(
            q=q,
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            cu_seqlens_kv=torch.tensor(
                [0, PAGE_SIZE], dtype=torch.int32, device="cuda"
            ),
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=lengths,
            is_causal=True,
            max_seqlen_q=1,
            max_seqlen_k=PAGE_SIZE,
            **block_scales,
        )
