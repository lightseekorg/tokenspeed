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
"""The fused index scoring + top-k kernel against separate scoring and top-k.

The fused kernel ranks 24-bit score codes, so it may swap keys whose scores
tie at the k-th place to that precision; every other selected key must be in
the exact top-k of the DeepGEMM scores.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

platform = current_platform()
if not (torch.cuda.is_available() and platform.is_nvidia and platform.is_blackwell):
    pytest.skip(
        "the fused DSA top-k needs an SM100-family GPU", allow_module_level=True
    )

from tokenspeed_kernel._optional import has_optional_package  # noqa: E402

if not has_optional_package("deep_gemm"):
    pytest.skip("the fused DSA top-k builds against deep_gemm", allow_module_level=True)

import deep_gemm  # noqa: E402
from tokenspeed_kernel.ops.attention.dsa import dsa_prefill_topk  # noqa: E402
from tokenspeed_kernel.ops.attention.dsa.flashinfer import (  # noqa: E402
    has_deterministic_decode_topk,
)
from tokenspeed_kernel.thirdparty.cuda import fused_mqa_topk  # noqa: E402

_HEADS = 32
_DIM = 128
_TOPK = 2048
# Two requests in one chunk; the second ranks more than min_keys(2048) keys.
_FIRST_KEYS = 40_000
_SECOND_KEYS = 70_000
_KEYS = _FIRST_KEYS + _SECOND_KEYS
# Relative score distance within which two keys tie for the fused kernel.
_TIE = 1e-3


def _ranges(rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal key ranges: half the rows on each request, one row with fewer
    keys than top-k and one with none."""
    half = rows // 2
    step = torch.arange(half, dtype=torch.int32, device="cuda")
    starts = torch.cat([torch.zeros_like(step), torch.full_like(step, _FIRST_KEYS)])
    ends = torch.cat([_FIRST_KEYS - half + step + 1, _KEYS - half + step + 1])
    ends[0] = 1000
    ends[1] = 0
    return starts.contiguous(), ends.to(torch.int32).contiguous()


def _inputs(rows: int):
    torch.manual_seed(0)
    q = (torch.randn(rows, _HEADS, _DIM, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    k = (torch.randn(_KEYS, _DIM, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    # Four equal UE8M0 bytes per key: exponents 126..128, scales 0.5, 1 and 2.
    exponent = torch.randint(126, 129, (_KEYS,), dtype=torch.int32, device="cuda")
    scale_words = exponent * 0x01010101
    weights = torch.rand(rows, _HEADS, device="cuda") + 0.1
    return q, k, scale_words, weights, *_ranges(rows)


def _exact_scores(q, k, scale_words, weights, starts, ends, row: int) -> torch.Tensor:
    unit = torch.full((1, _HEADS), 0x7F7F7F7F, dtype=torch.int32, device="cuda")
    count = int(ends[row] - starts[row])
    return deep_gemm.fp8_fp4_mqa_logits(
        (q[row : row + 1].contiguous(), unit),
        (k, scale_words),
        weights[row : row + 1].contiguous(),
        starts[row : row + 1],
        ends[row : row + 1],
        max_seqlen_k=count,
    )[0, :count].float()


@pytest.mark.parametrize("rows", [2048, fused_mqa_topk.MAX_ROWS + 1200])
def test_fused_topk_selects_the_exact_topk_up_to_ties(rows: int) -> None:
    q, k, scale_words, weights, starts, ends = _inputs(rows)
    out = torch.empty(rows, _TOPK, dtype=torch.int32, device="cuda")
    fused_mqa_topk.fused_mqa_topk_prefill(
        q, k, fused_mqa_topk.key_scales(scale_words), weights, starts, ends, out
    )
    lengths = (ends - starts).clamp_min(0)

    assert out[1].eq(-1).all()
    short = out[0]
    assert torch.equal(short[:1000], torch.arange(1000, device="cuda").int())
    assert short[1000:].eq(-1).all()

    checked = torch.linspace(2, rows - 1, 48, device="cuda").long().tolist()
    for row in checked + [rows // 2, min(fused_mqa_topk.MAX_ROWS, rows - 1)]:
        length = int(lengths[row])
        selected = out[row]
        assert selected.min() >= 0 and selected.max() < length
        assert selected.unique().numel() == _TOPK
        scores = _exact_scores(q, k, scale_words, weights, starts, ends, row)
        kth = scores.topk(_TOPK).values[-1]
        # Every selected key reaches the k-th score, up to a tie.
        assert scores[selected.long()].min() >= kth - _TIE * kth.abs()


def test_prefill_topk_takes_the_fused_kernel_and_matches_the_dense_path() -> None:
    if not has_deterministic_decode_topk():
        pytest.skip("needs the lowest-index tie-break top-k as the reference")
    rows = 2048
    torch.manual_seed(1)
    q = torch.randn(rows, _HEADS, _DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(rows, _HEADS, device="cuda") + 0.1
    keys = torch.randn(_KEYS, _DIM, device="cuda").to(torch.float8_e4m3fn)
    scale_words = torch.full((_KEYS, 1), 0x7F7F7F7F, dtype=torch.int32, device="cuda")
    starts, ends = _ranges(rows)
    lengths = (ends - starts).clamp_min(0)

    def select(batch_invariant: bool) -> torch.Tensor:
        selected, _ = dsa_prefill_topk(
            q,
            weights,
            torch.arange(_KEYS, dtype=torch.int64, device="cuda"),
            starts,
            ends,
            topk=_TOPK,
            softmax_scale=1.0,
            batch_invariant=batch_invariant,
            index_k_fp8=keys,
            index_k_scale=scale_words,
            max_logits_bytes=1 << 30,
            candidate_lens_cpu=lengths.cpu().long(),
            initial_tokens=0,
            local_tokens=0,
            slot_order="selection",
            solution="deep_gemm",
        )
        return selected

    assert fused_mqa_topk.beats_dense(
        q.device,
        rows=rows,
        dense_rows=(1 << 30) // (_KEYS * 4),
        keys=_KEYS,
        max_row_keys=int(lengths.max()),
        topk=_TOPK,
    )
    fused = select(batch_invariant=False)
    dense = select(batch_invariant=True)
    assert fused[1].eq(-1).all() and dense[1].eq(-1).all()
    shared = 0
    for row in range(2, rows, 61):
        got, want = set(fused[row].tolist()), set(dense[row].tolist())
        assert min(got) >= int(starts[row]) and max(got) < int(ends[row])
        shared += len(got & want)
        # Ties at the k-th score are the only keys the two paths may swap.
        assert len(got & want) >= _TOPK - 8
    assert shared >= 0.999 * _TOPK * len(range(2, rows, 61))


def test_beats_dense_follows_the_measured_thresholds() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    full = dict(rows=8 * sms, dense_rows=8 * sms, keys=200_000, max_row_keys=200_000)
    assert fused_mqa_topk.beats_dense(device, topk=2048, **full)
    assert not fused_mqa_topk.beats_dense(
        device, topk=2048, **{**full, "max_row_keys": 60_000}
    )
    assert fused_mqa_topk.beats_dense(
        device, topk=1024, **{**full, "max_row_keys": 60_000}
    )
    assert not fused_mqa_topk.beats_dense(
        device, topk=2048, **{**full, "keys": fused_mqa_topk.MAX_KEYS + 1}
    )
    # A last wave on a quarter of the SMs loses, unless dense needs two launches.
    quarter = {**full, "rows": 8 * (sms + sms // 4)}
    assert not fused_mqa_topk.beats_dense(
        device, topk=2048, **{**quarter, "dense_rows": quarter["rows"]}
    )
    assert fused_mqa_topk.beats_dense(device, topk=2048, **quarter)
