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

"""The skip decision is per row, but only a unanimous block changes anything.

On a block where at least one row dissents, the per-row votes are discarded and
every row takes the ordinary online-softmax update. That is the contract pinned
here, and the easy one to get wrong: dropping a dissenting block for the rows
that did vote is both less accurate and slower, since holding a row's max back
makes it less likely to vote on later blocks. The authors' own Hopper kernel
in TensorRT-LLM discards the per-row bits the same way.

Checks (bf16, causal, fixed seed):
  [1] VOTES ARE INERT   thresholds where many rows vote but no block is
      unanimous give output bit-identical to a threshold too small for any
      row to vote, across GQA ratios, sinks, LSE, sliding window and ragged
      batch
  [2] ELISION FIRES     past that range the output does move, so [1] is not
      vacuously true of a kernel that skips nothing
  [3] NO REGRESSION     with skipping off, the result matches dense SDPA

How far the elided output may drift is covered by
``test_mha_prefill_skip_softmax.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel_amd.ops.gfx950.attention.mha.prefill import (
    gluon_mha_prefill_gfx950,
)
from utils import is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for Gluon skip-softmax MHA prefill tests",
        allow_module_level=True,
    )


_SEQLEN = 4096
_HEAD_DIM = 128
_DTYPE = torch.bfloat16
_NO_REGRESSION_TOL = 5e-3
# Baseline for [1]: too small for any row to vote, but still the
# ENABLE_SKIP_SOFTMAX=True build, so any difference is the skipping itself.
_TINY_THRESHOLD = 1e-9
# Rows vote, no block goes unanimous. Measured at 8/2 heads, seqlen 4096,
# seed 0: 0.3 has 31.9% of row-block pairs voting and 0% of blocks elided.
_ROW_VOTE_RATES = [1e-3, 1e-2, 1e-1, 0.3]
# Past that range: 1.16% of blocks elided at 0.9 on the same shape.
_ELIDING_THRESHOLDS = [0.7, 0.9]
_GQA_SHAPES = [(8, 8), (8, 2)]


def _qkv(n_heads: int, n_kv_heads: int, total_tokens: int, seed: int = 0):
    torch.manual_seed(seed)
    q = torch.randn((total_tokens, n_heads, _HEAD_DIM), device="cuda", dtype=_DTYPE)
    k = torch.randn((total_tokens, n_kv_heads, _HEAD_DIM), device="cuda", dtype=_DTYPE)
    v = torch.randn((total_tokens, n_kv_heads, _HEAD_DIM), device="cuda", dtype=_DTYPE)
    return q, k, v


def _dense_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Per-sequence causal dense reference in fp32. [S,H,D] layout, one sequence."""
    n_heads, n_kv_heads = q.shape[1], k.shape[1]
    qt, kt, vt = (x.transpose(0, 1).float().unsqueeze(0) for x in (q, k, v))
    if n_kv_heads != n_heads:
        repeat = n_heads // n_kv_heads
        kt = kt.repeat_interleave(repeat, dim=1)
        vt = vt.repeat_interleave(repeat, dim=1)
    out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
    return out.squeeze(0).transpose(0, 1).to(q.dtype)


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (
        (a.float() - b.float()).abs().mean() / b.float().abs().mean().clamp_min(1e-6)
    ).item()


def _run(q, k, v, seqlens: list[int], threshold: float, **kwargs):
    cu = [0]
    for s in seqlens:
        cu.append(cu[-1] + s)
    return gluon_mha_prefill_gfx950(
        q=q,
        k=k,
        v=v,
        cu_seqlens=torch.tensor(cu, device="cuda", dtype=torch.int32),
        cu_seqlens_cpu=cu,
        max_seqlen=max(seqlens),
        skip_softmax_threshold=threshold,
        **kwargs,
    )


@pytest.mark.parametrize("n_heads,n_kv_heads", _GQA_SHAPES)
@pytest.mark.parametrize("threshold", _ROW_VOTE_RATES)
def test_row_votes_alone_change_nothing(
    n_heads: int, n_kv_heads: int, threshold: float
) -> None:
    """[1] Rows vote, but a block that is not unanimous must be exact."""
    q, k, v = _qkv(n_heads, n_kv_heads, _SEQLEN)
    voting = _run(q, k, v, [_SEQLEN], threshold)
    inert = _run(q, k, v, [_SEQLEN], _TINY_THRESHOLD)
    assert torch.isfinite(voting).all()
    assert torch.equal(voting, inert)


@pytest.mark.parametrize("threshold", _ROW_VOTE_RATES)
def test_row_votes_alone_preserve_lse(threshold: float) -> None:
    """[1] l_i and m_i are what a mishandled vote would corrupt first."""
    q, k, v = _qkv(8, 2, _SEQLEN)
    o_voting, lse_voting = _run(q, k, v, [_SEQLEN], threshold, return_lse=True)
    o_inert, lse_inert = _run(q, k, v, [_SEQLEN], _TINY_THRESHOLD, return_lse=True)
    assert torch.isfinite(lse_voting).all()
    assert torch.equal(o_voting, o_inert)
    assert torch.equal(lse_voting, lse_inert)


@pytest.mark.parametrize("threshold", _ROW_VOTE_RATES)
def test_row_votes_alone_with_sinks(threshold: float) -> None:
    """[1] Sinks change the m_i initialization, which the vote reads."""
    q, k, v = _qkv(8, 2, _SEQLEN)
    sinks = torch.randn((8,), device="cuda", dtype=torch.float32)
    voting = _run(q, k, v, [_SEQLEN], threshold, sinks=sinks)
    inert = _run(q, k, v, [_SEQLEN], _TINY_THRESHOLD, sinks=sinks)
    assert torch.isfinite(voting).all()
    assert torch.equal(voting, inert)


@pytest.mark.parametrize("threshold", _ROW_VOTE_RATES)
def test_row_votes_alone_sliding_window(threshold: float) -> None:
    """[1] The HAS_INVALID variant, where a fully masked tile is unanimous.

    Every row's max is -inf there, so every row votes, and eliding is correct
    because the tile contributes zero.
    """
    q, k, v = _qkv(8, 8, _SEQLEN)
    voting = _run(q, k, v, [_SEQLEN], threshold, window_left=256)
    inert = _run(q, k, v, [_SEQLEN], _TINY_THRESHOLD, window_left=256)
    assert torch.isfinite(voting).all()
    assert torch.equal(voting, inert)


@pytest.mark.parametrize("threshold", _ROW_VOTE_RATES)
def test_row_votes_alone_ragged_batch(threshold: float) -> None:
    """[1] Includes a sequence shorter than BLOCK_M, which takes its own path."""
    seqlens = [1024, 64, 2048, 512]
    q, k, v = _qkv(8, 2, sum(seqlens), seed=1)
    voting = _run(q, k, v, seqlens, threshold)
    inert = _run(q, k, v, seqlens, _TINY_THRESHOLD)
    assert torch.isfinite(voting).all()
    assert torch.equal(voting, inert)


@pytest.mark.parametrize("threshold", _ELIDING_THRESHOLDS)
def test_elision_actually_fires(threshold: float) -> None:
    """[2] Blocks do go unanimous past the vote-only range, and output moves."""
    q, k, v = _qkv(8, 2, _SEQLEN)
    elided = _run(q, k, v, [_SEQLEN], threshold)
    inert = _run(q, k, v, [_SEQLEN], _TINY_THRESHOLD)
    assert torch.isfinite(elided).all()
    assert not torch.equal(elided, inert)


@pytest.mark.parametrize("threshold", [0.0, _TINY_THRESHOLD])
def test_inert_without_skipping(threshold: float) -> None:
    """[3] With nothing to vote or elide, the result still matches dense."""
    q, k, v = _qkv(8, 2, _SEQLEN)
    out = _run(q, k, v, [_SEQLEN], threshold)
    assert torch.isfinite(out).all()
    assert _rel_err(out, _dense_ref(q, k, v)) < _NO_REGRESSION_TOL
