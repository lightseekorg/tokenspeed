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

"""Tests for the DSA decode indexer top-k backends.

Covers the per-solution wrappers behind ``deep_gemm_dsa_decode_topk``:
  * ``ragged_decode_topk`` (CUDA persistent-radix) dispatch + arg forwarding.
  * ``deterministic_decode_topk`` (flashinfer) pre-masked fallback path.
  * ``cute_dsl_decode_topk`` (CuTe DSL cluster radix): per-row causal-window
    ``torch.topk`` accuracy across batch / next_n / top-k / context length /
    compression ratio, plus row-aligned (widened) logits equivalence.
  * ``combine_topk_weights`` (Triton): bit-exactness against the unfused
    ``weights.float() * q_scale * softmax_scale`` reference chain.
  * ``_prepare_logits_for_topk``: widening of DeepGEMM's narrowed logits
    slices plus the in-place NaN/inf -> -inf scrub.

The wrapper-dispatch tests are CPU-only (monkeypatched kernels); the CuTe DSL
kernel test needs NVIDIA Blackwell (sm_100+) and skips elsewhere.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.attention.cuda import dsa_topk as cuda_dsa_topk
from tokenspeed_kernel.ops.attention.cute_dsl.dsa_topk import (
    cute_dsl_decode_topk,
    has_cute_dsl_decode_topk,
)
from tokenspeed_kernel.ops.attention.deep_gemm import _prepare_logits_for_topk
from tokenspeed_kernel.ops.attention.flashinfer import dsa_topk as fi_dsa_topk
from tokenspeed_kernel.ops.attention.triton.dsa_topk import combine_topk_weights

requires_kernel = pytest.mark.skipif(
    not (torch.cuda.is_available() and has_cute_dsl_decode_topk()),
    reason="CuTe DSL DSA decode top-k requires NVIDIA Blackwell (sm_100+)",
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


# ---------------------------------------------------------------------------
# Wrapper dispatch (CPU, monkeypatched kernels)
# ---------------------------------------------------------------------------
def test_ragged_decode_topk_delegates_to_persistent_topk(monkeypatch):
    calls = {}

    monkeypatch.setattr(cuda_dsa_topk, "has_persistent_topk", lambda: True)

    def fake_persistent_topk(
        logits, lengths, output, workspace, k, max_seq_len, q_len_per_req=1
    ):
        calls["logits"] = logits
        calls["lengths"] = lengths
        calls["workspace"] = workspace
        calls["k"] = k
        calls["max_seq_len"] = max_seq_len
        calls["q_len_per_req"] = q_len_per_req
        output.fill_(7)

    monkeypatch.setattr(cuda_dsa_topk, "persistent_topk", fake_persistent_topk)

    logits = torch.randn(2, 8, dtype=torch.float32)
    lengths = torch.tensor([[3], [6]], dtype=torch.int64)
    output = torch.empty(2, 4, dtype=torch.int32)
    workspace = torch.empty(32, dtype=torch.uint8)

    cuda_dsa_topk.ragged_decode_topk(
        logits,
        output,
        4,
        lengths=lengths,
        workspace=workspace,
        max_seq_len=8,
    )

    assert torch.equal(output, torch.full_like(output, 7))
    assert calls["logits"].is_contiguous()
    assert torch.equal(calls["lengths"], torch.tensor([3, 6], dtype=torch.int32))
    assert calls["workspace"] is workspace
    assert calls["k"] == 4
    assert calls["max_seq_len"] == 8
    assert calls["q_len_per_req"] == 1


def test_ragged_decode_topk_raises_when_persistent_kernel_unavailable(monkeypatch):
    monkeypatch.setattr(cuda_dsa_topk, "has_persistent_topk", lambda: False)

    logits = torch.randn(2, 8, dtype=torch.float32)
    output = torch.empty(2, 4, dtype=torch.int32)
    lengths = torch.tensor([3, 6], dtype=torch.int32)
    workspace = torch.empty(32, dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="length-aware"):
        cuda_dsa_topk.ragged_decode_topk(
            logits,
            output,
            4,
            lengths=lengths,
            workspace=workspace,
            max_seq_len=8,
        )


def test_deterministic_decode_topk_falls_back_to_flashinfer(monkeypatch):
    calls = {}
    indices = torch.tensor([[1, 0, 3], [2, 4, 1]], dtype=torch.int64)

    def fake_top_k(logits, k, *, deterministic, tie_break, dsa_graph_safe):
        calls["logits"] = logits
        calls["k"] = k
        calls["deterministic"] = deterministic
        calls["tie_break"] = tie_break
        calls["dsa_graph_safe"] = dsa_graph_safe
        return None, indices

    monkeypatch.setattr(fi_dsa_topk, "top_k", fake_top_k)
    monkeypatch.setattr(fi_dsa_topk, "TopKTieBreak", SimpleNamespace(SMALL="small"))

    logits = torch.randn(2, 8, dtype=torch.float32)
    output = torch.empty(2, 3, dtype=torch.int32)

    fi_dsa_topk.deterministic_decode_topk(logits, output, 3)

    assert torch.equal(output, indices.to(torch.int32))
    assert calls["logits"].is_contiguous()
    assert calls["k"] == 3
    assert calls["deterministic"] is True
    assert calls["tie_break"] == "small"
    assert calls["dsa_graph_safe"] is True


# ---------------------------------------------------------------------------
# CuTe DSL single-pass multi-CTA (cluster) kernel (NVIDIA Blackwell sm_100+)
# ---------------------------------------------------------------------------
def _row_window(seq_lens: torch.Tensor, row: int, next_n: int, num_cols: int) -> int:
    """Causal candidate window for output row ``row`` (see kernel contract)."""
    req = row // next_n
    win = int(seq_lens[req]) - next_n + (row % next_n) + 1
    return max(0, min(win, num_cols))


def _reference_topk_values(
    logits: torch.Tensor, seq_lens: torch.Tensor, topk: int, next_n: int
) -> torch.Tensor:
    """Per-row causal-window top-k values, sorted ascending, ``-inf`` padded."""
    num_rows, num_cols = logits.shape
    out = torch.full((num_rows, topk), float("-inf"))
    for r in range(num_rows):
        win = _row_window(seq_lens, r, next_n, num_cols)
        k = min(topk, win)
        if k > 0:
            vals = logits[r, :win].topk(k).values.sort().values
            out[r, :k] = vals.cpu()
    return out


def _gathered_topk_values(
    logits: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    topk: int,
    next_n: int,
) -> torch.Tensor:
    """Values selected by ``indices``, per row, sorted ascending, ``-inf`` pad."""
    num_rows, num_cols = logits.shape
    out = torch.full((num_rows, topk), float("-inf"))
    for r in range(num_rows):
        win = _row_window(seq_lens, r, next_n, num_cols)
        k = min(topk, win)
        if k > 0:
            sel = indices[r, :k].long()
            # Every selected index must be inside the causal window.
            assert (sel >= 0).all() and (
                sel < win
            ).all(), f"row {r}: index outside causal window [0,{win})"
            out[r, :k] = logits[r].gather(0, sel).sort().values.cpu()
    return out


@requires_kernel
@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 64])
@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("index_topk", [2048, 512, 128])
@pytest.mark.parametrize("num_tokens", [4096, 8192, 16384, 32768, 65536, 131072])
@pytest.mark.parametrize("compress_ratio", [1, 4])
def test_cute_dsl_decode_topk(
    batch_size, next_n, index_topk, num_tokens, compress_ratio
):
    """cute_dsl_decode_topk selects the correct per-row causal-window top-k.

    ``num_tokens`` is the raw context length; the indexer selects over the
    compressed candidate columns ``ceil(num_tokens / compress_ratio)``. Each
    row's causal window (in compressed units) is derived from ``seq_lens``, and
    the selected columns must gather exactly the reference top-k values.
    """
    num_rows = batch_size * next_n
    num_cols = -(-num_tokens // compress_ratio)  # ceil(num_tokens / compress_ratio)
    torch.manual_seed(0)
    logits = torch.randn(num_rows, num_cols, device="cuda", dtype=torch.float32)
    # Per-request compressed lengths straddle top_k so both the window < top_k
    # and window >= top_k regimes are exercised.
    low = max(1, min(index_topk // 4, num_cols))
    seq_lens = torch.randint(
        low, num_cols + 1, (batch_size,), device="cuda", dtype=torch.int32
    )
    out = torch.empty(num_rows, index_topk, device="cuda", dtype=torch.int32)

    ret = cute_dsl_decode_topk(logits, seq_lens, index_topk, next_n=next_n, out=out)

    assert ret.data_ptr() == out.data_ptr(), "out must be written in place"
    assert ret.dtype == torch.int32 and ret.shape == (num_rows, index_topk)

    got = _gathered_topk_values(logits, ret, seq_lens, index_topk, next_n)
    ref = _reference_topk_values(logits, seq_lens, index_topk, next_n)
    assert torch.equal(got, ref), "selected top-k values differ from reference"


@requires_kernel
@pytest.mark.parametrize("next_n", [1, 2])
def test_cute_dsl_decode_topk_widened_rows_match_narrow(next_n):
    """Row-aligned (widened) logits select identically to the narrow slice.

    DeepGEMM row-aligns the logits allocation; ``deep_gemm_dsa_decode_topk``
    hands the widened compact view straight to the kernel instead of paying a
    ``.contiguous()`` copy of the narrow slice. The kernel is length-aware, so
    the padding columns -- poisoned with NaN here -- must never influence the
    selection.
    """
    batch_size, num_cols, aligned_cols, index_topk = 4, 4992, 5120, 2048
    num_rows = batch_size * next_n
    torch.manual_seed(1)
    buf = torch.full(
        (num_rows, aligned_cols), float("nan"), device="cuda", dtype=torch.float32
    )
    buf[:, :num_cols].normal_()
    narrow = buf[:, :num_cols]
    assert not narrow.is_contiguous()
    seq_lens = torch.randint(
        index_topk // 4, num_cols + 1, (batch_size,), device="cuda", dtype=torch.int32
    )

    out_narrow = torch.empty(num_rows, index_topk, device="cuda", dtype=torch.int32)
    cute_dsl_decode_topk(
        narrow.contiguous(), seq_lens, index_topk, next_n=next_n, out=out_narrow
    )
    out_wide = torch.empty(num_rows, index_topk, device="cuda", dtype=torch.int32)
    cute_dsl_decode_topk(buf, seq_lens, index_topk, next_n=next_n, out=out_wide)

    got = _gathered_topk_values(narrow, out_wide, seq_lens, index_topk, next_n)
    ref = _gathered_topk_values(narrow, out_narrow, seq_lens, index_topk, next_n)
    assert torch.equal(got, ref), "widened-view selection differs from narrow slice"


# ---------------------------------------------------------------------------
# combine_topk_weights (Triton fused weights * q_scale * softmax_scale)
# ---------------------------------------------------------------------------
@requires_cuda
@pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("split_view", [False, True])
@pytest.mark.parametrize("tokens,heads", [(1, 64), (16, 64), (37, 64), (128, 32)])
def test_combine_topk_weights_bitexact(weights_dtype, split_view, tokens, heads):
    """Fused combine is bit-exact with the unfused fp32 reference chain.

    ``split_view=True`` mirrors production: ``weights`` is the trailing
    column-``split`` view of the fused ``wk_weights_proj`` output, so its rows
    are unit-stride but not compact.
    """
    torch.manual_seed(tokens * heads)
    softmax_scale = 0.08838834764831845 * (64**-0.5)
    if split_view:
        fused = torch.randn(tokens, 128 + heads, device="cuda", dtype=weights_dtype)
        weights = fused.split([128, heads], dim=-1)[1]
        # (a [1, heads] slice is trivially "contiguous"; multi-row ones are not)
        assert weights.stride(-1) == 1
        assert tokens == 1 or not weights.is_contiguous()
    else:
        weights = torch.randn(tokens, heads, device="cuda", dtype=weights_dtype)
    q_scale = torch.rand(tokens * heads, 1, device="cuda", dtype=torch.float32) + 0.01

    ref = (
        weights.float().unsqueeze(-1)
        * q_scale.view(tokens, heads, 1)
        * float(softmax_scale)
    ).squeeze(-1)
    got = combine_topk_weights(weights, q_scale, softmax_scale)

    assert got.dtype == torch.float32 and got.shape == (tokens, heads)
    assert got.is_contiguous()
    assert torch.equal(got, ref)


@requires_cuda
def test_combine_topk_weights_empty():
    out = combine_topk_weights(
        torch.empty(0, 64, device="cuda", dtype=torch.bfloat16),
        torch.empty(0, 1, device="cuda", dtype=torch.float32),
        0.5,
    )
    assert out.shape == (0, 64) and out.dtype == torch.float32


# ---------------------------------------------------------------------------
# _prepare_logits_for_topk (DeepGEMM narrowed-logits widening + NaN/inf scrub)
# ---------------------------------------------------------------------------
def test_prepare_logits_for_topk_widens_and_scrubs():
    base = torch.arange(4 * 256, dtype=torch.float32).view(4, 256)
    expected = base.clone()
    # Non-finite values inside the window and in the padding columns alike.
    base[0, 10] = float("nan")
    base[1, 20] = float("inf")
    base[2, 200] = float("-inf")  # padding column (>= 192)
    expected[0, 10] = expected[1, 20] = expected[2, 200] = float("-inf")
    narrow = base[:, :192]
    assert not narrow.is_contiguous()

    full = _prepare_logits_for_topk(narrow)

    assert full.shape == (4, 256) and full.is_contiguous()
    assert torch.equal(full, expected)
    # The scrub is in place: the narrow slice aliases the cleaned storage.
    assert narrow[0, 10] == float("-inf") and narrow[1, 20] == float("-inf")
    # In-place edits through the widened view alias the narrow slice.
    full[1, 0] = -1.0
    assert narrow[1, 0] == -1.0


def test_prepare_logits_for_topk_passthrough_still_scrubs():
    # Already-compact logits: same object back, values scrubbed in place.
    compact = torch.zeros(4, 256)
    compact[3, 5] = float("nan")
    assert _prepare_logits_for_topk(compact) is compact
    assert compact[3, 5] == float("-inf")
    # Column slices don't own their full rows: no widening, scrub only the
    # slice (the storage outside the view must stay untouched).
    owner = torch.full((4, 256), float("nan"))
    col_slice = owner[:, 64:192]
    assert _prepare_logits_for_topk(col_slice) is col_slice
    assert (col_slice == float("-inf")).all()
    assert owner[:, :64].isnan().all() and owner[:, 192:].isnan().all()
    # Non-unit column stride: no widening, scrubbed in place.
    strided = torch.full((4, 256), float("inf"))[:, ::2]
    assert _prepare_logits_for_topk(strided) is strided
    assert (strided == float("-inf")).all()
    # 3-D tensors: no widening, scrubbed in place.
    cube = torch.full((2, 3, 4), float("nan"))
    assert _prepare_logits_for_topk(cube) is cube
    assert (cube == float("-inf")).all()
