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

"""The work counter decides placement at runtime; the result must not depend
on it.

A nonzero ``skip_softmax_threshold`` switches the persistent scheduler to a
global atomic ticket counter, so which workgroup runs which (batch, head,
query block) varies between runs of the same launch. Each item writes a
disjoint output slice.

Checks (bf16, causal):
  [1] COVERAGE     matches dense SDPA at a threshold too small to skip, so a
                   dropped or duplicated ticket cannot pass as sparsity
  [2] DETERMINISM  repeated launches are bit-identical
  [3] SHAPES       GQA ratios 1, 2, 4 and 8, the group sizes the ticket
                   decode interleaves over
  [4] RAGGED       mixed lengths, one below BLOCK_M, covering the padded
                   tail of the ticket space
  [5] SLIDING      the sliding kernel keeps the static order

No speedup is asserted: the benefit depends on sparsity being unevenly
distributed across heads, which random Q/K/V does not reproduce.
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
        "AMD CDNA4 is required for Gluon dynamic-schedule MHA prefill tests",
        allow_module_level=True,
    )


_SEQLEN = 4096
_HEAD_DIM = 128
_DTYPE = torch.bfloat16
_NO_REGRESSION_TOL = 5e-3
# Too small to skip anything, so [1] measures scheduling and not sparsity.
_TINY_THRESHOLD = 1e-9
_THRESHOLDS = [1e-3, 1e-2, 1e-1]
# (n_heads, n_kv_heads): group sizes 1, 2, 4 and 8.
_GQA_SHAPES = [(8, 8), (8, 4), (8, 2), (8, 1)]
_REPEATS = 5


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
@pytest.mark.parametrize("defer_v_load", [False, True])
def test_dynamic_sched_covers_every_row(
    n_heads: int, n_kv_heads: int, defer_v_load: bool
) -> None:
    """[1][3] Dropped or duplicated tickets leave rows wrong; dense catches it."""
    q, k, v = _qkv(n_heads, n_kv_heads, _SEQLEN)
    out = _run(q, k, v, [_SEQLEN], _TINY_THRESHOLD, defer_v_load=defer_v_load)
    assert torch.isfinite(out).all()
    assert _rel_err(out, _dense_ref(q, k, v)) < _NO_REGRESSION_TOL


@pytest.mark.parametrize("n_heads,n_kv_heads", _GQA_SHAPES)
@pytest.mark.parametrize("threshold", _THRESHOLDS)
def test_dynamic_sched_is_deterministic(
    n_heads: int, n_kv_heads: int, threshold: float
) -> None:
    """[2][3] Repeated, since a schedule-dependent result need not differ on
    any particular pair of launches."""
    q, k, v = _qkv(n_heads, n_kv_heads, _SEQLEN)
    first = _run(q, k, v, [_SEQLEN], threshold)
    for _ in range(_REPEATS - 1):
        assert torch.equal(_run(q, k, v, [_SEQLEN], threshold), first)


@pytest.mark.parametrize("threshold", _THRESHOLDS)
def test_dynamic_sched_ragged_batch(threshold: float) -> None:
    """[4] Padded tail of the ticket space, plus a sequence under BLOCK_M."""
    seqlens = [1024, 64, 2048, 512]
    n_heads, n_kv_heads = 8, 2
    q, k, v = _qkv(n_heads, n_kv_heads, sum(seqlens))
    out = _run(q, k, v, seqlens, _TINY_THRESHOLD)
    assert torch.isfinite(out).all()

    start = 0
    for s in seqlens:
        sl = slice(start, start + s)
        ref = _dense_ref(q[sl], k[sl], v[sl])
        assert _rel_err(out[sl], ref) < _NO_REGRESSION_TOL, f"seqlen {s}"
        start += s

    first = _run(q, k, v, seqlens, threshold)
    for _ in range(_REPEATS - 1):
        assert torch.equal(_run(q, k, v, seqlens, threshold), first)


@pytest.mark.parametrize("threshold", [0.0] + _THRESHOLDS)
def test_sliding_window_unaffected(threshold: float) -> None:
    """[5] The sliding kernel is never dynamically scheduled (see
    gluon_mha_prefill_gfx950: dynamic_sched is forced to 0 whenever
    is_sliding), so its output must not depend on the threshold beyond
    ordinary reduction-order noise, not just repeatable within one."""
    q, k, v = _qkv(8, 8, _SEQLEN)
    baseline = _run(q, k, v, [_SEQLEN], 0.0, window_left=256)
    assert torch.isfinite(baseline).all()
    for _ in range(_REPEATS):
        assert (
            _rel_err(_run(q, k, v, [_SEQLEN], threshold, window_left=256), baseline)
            < _NO_REGRESSION_TOL
        )
