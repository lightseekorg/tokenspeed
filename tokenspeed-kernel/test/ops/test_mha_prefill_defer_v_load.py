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

"""``defer_v_load`` reorders memory traffic and nothing else.

The main loop issues V's HBM->LDS load only after the skip decision is known,
so a fully skipped block costs no V traffic. Which blocks are skipped does not
change, so the output must not either.

Checks (bf16, causal):
  [1] EQUIVALENCE    bit-identical to ``defer_v_load=False`` across the
      threshold range, with sinks, a sliding window and LSE
  [2] SHARED CODE    the skip-softmax checks from
      ``test_mha_prefill_skip_softmax.py``, rerun with the flag on, since
      [1] cancels any defect that hits both paths equally
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
        "AMD CDNA4 is required for Gluon defer_v_load MHA prefill tests",
        allow_module_level=True,
    )


_SEQLEN = 4096
_NUM_Q_HEADS = 8
_NUM_KV_HEADS = 8
_HEAD_DIM = 128
_DTYPE = torch.bfloat16
_NO_REGRESSION_TOL = 5e-3
_DEGRADATION_BOUND = 0.6
# See the note on the same constant in ``test_mha_prefill_skip_softmax.py``.
_THRESHOLDS = [1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 0.7, 0.9]
_TINY_THRESHOLD = 1e-9


def _qkv(seed: int = 0):
    torch.manual_seed(seed)
    shape = (_SEQLEN, _NUM_Q_HEADS, _HEAD_DIM)
    kv_shape = (_SEQLEN, _NUM_KV_HEADS, _HEAD_DIM)
    q = torch.randn(shape, device="cuda", dtype=_DTYPE)
    k = torch.randn(kv_shape, device="cuda", dtype=_DTYPE)
    v = torch.randn(kv_shape, device="cuda", dtype=_DTYPE)
    return q, k, v


def _dense_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Single-sequence causal dense reference; fp32 SDPA math. [S,H,D] layout."""
    qt, kt, vt = (x.transpose(0, 1).float().unsqueeze(0) for x in (q, k, v))
    out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
    return out.squeeze(0).transpose(0, 1).to(q.dtype)


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (
        (a.float() - b.float()).abs().mean() / b.float().abs().mean().clamp_min(1e-6)
    ).item()


def _cu_seqlens():
    cu = torch.tensor([0, _SEQLEN], device="cuda", dtype=torch.int32)
    return cu, [0, _SEQLEN]


def _run(q, k, v, skip_softmax_threshold: float, defer_v_load: bool = True, **kwargs):
    cu_seqlens, cu_seqlens_cpu = _cu_seqlens()
    return gluon_mha_prefill_gfx950(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        max_seqlen=_SEQLEN,
        skip_softmax_threshold=skip_softmax_threshold,
        defer_v_load=defer_v_load,
        **kwargs,
    )


def test_defer_v_load_no_regression() -> None:
    q, k, v = _qkv()
    dense = _dense_ref(q, k, v)
    out0 = _run(q, k, v, skip_softmax_threshold=0.0)
    assert torch.isfinite(out0).all()
    assert _rel_err(out0, dense) < _NO_REGRESSION_TOL


def test_defer_v_load_tiny_threshold_matches_dense() -> None:
    q, k, v = _qkv()
    dense = _dense_ref(q, k, v)
    out = _run(q, k, v, skip_softmax_threshold=_TINY_THRESHOLD)
    assert torch.isfinite(out).all()
    assert _rel_err(out, dense) < _NO_REGRESSION_TOL


@pytest.mark.parametrize("threshold", _THRESHOLDS)
def test_defer_v_load_degrades_gracefully(threshold: float) -> None:
    q, k, v = _qkv()
    dense = _dense_ref(q, k, v)
    out = _run(q, k, v, skip_softmax_threshold=threshold)
    assert torch.isfinite(out).all()
    assert _rel_err(out, dense) < _DEGRADATION_BOUND


def test_defer_v_load_actually_skips() -> None:
    q, k, v = _qkv()
    dense = _dense_ref(q, k, v)
    r_lo = _rel_err(_run(q, k, v, skip_softmax_threshold=_THRESHOLDS[0]), dense)
    r_hi = _rel_err(_run(q, k, v, skip_softmax_threshold=_THRESHOLDS[-1]), dense)
    assert r_hi > r_lo


@pytest.mark.parametrize("threshold", [2.0, 5.0, 12.0])
def test_defer_v_load_high_threshold_no_nan(threshold: float) -> None:
    """[2] log2_threshold > 0 must stay finite on the deferred path too."""
    q, k, v = _qkv()
    out = _run(q, k, v, skip_softmax_threshold=threshold)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "threshold", [0.0, _TINY_THRESHOLD, *_THRESHOLDS, 2.0, 5.0, 12.0]
)
def test_defer_v_load_matches_co_issue_exactly(threshold: float) -> None:
    """[1] Bit-identical, which the dense-SDPA tolerances above cannot pin.

    Catches defects confined to the deferred branch: where issue_load_v sits
    relative to the skip decision, and its wait_group depth.
    """
    q, k, v = _qkv()
    deferred = _run(q, k, v, skip_softmax_threshold=threshold, defer_v_load=True)
    co_issue = _run(q, k, v, skip_softmax_threshold=threshold, defer_v_load=False)
    assert torch.equal(deferred, co_issue)


@pytest.mark.parametrize("threshold", [0.0, 3e-1, 12.0])
@pytest.mark.parametrize("feature", ["sinks", "sliding"])
def test_defer_v_load_matches_co_issue_with_other_features(
    threshold: float, feature: str
) -> None:
    """[1] Sinks change the initial running max; sliding forces the flag off."""
    q, k, v = _qkv()
    if feature == "sinks":
        kwargs = {
            "sinks": torch.randn((_NUM_Q_HEADS,), device="cuda", dtype=torch.float32)
        }
    else:
        kwargs = {"window_left": 256}
    deferred = _run(q, k, v, threshold, defer_v_load=True, **kwargs)
    co_issue = _run(q, k, v, threshold, defer_v_load=False, **kwargs)
    assert torch.equal(deferred, co_issue)


@pytest.mark.parametrize("threshold", [0.0, 3e-1, 12.0])
def test_defer_v_load_matches_co_issue_with_lse(threshold: float) -> None:
    """Deferring V's load must not perturb the returned LSE either."""
    q, k, v = _qkv()
    out_d, lse_d = _run(q, k, v, threshold, defer_v_load=True, return_lse=True)
    out_c, lse_c = _run(q, k, v, threshold, defer_v_load=False, return_lse=True)
    assert torch.equal(out_d, out_c)
    assert torch.equal(lse_d, lse_c)
