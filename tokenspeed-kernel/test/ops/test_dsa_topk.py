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
    compression ratio.

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
from tokenspeed_kernel.ops.attention.flashinfer import dsa_topk as fi_dsa_topk

requires_kernel = pytest.mark.skipif(
    not (torch.cuda.is_available() and has_cute_dsl_decode_topk()),
    reason="CuTe DSL DSA decode top-k requires NVIDIA Blackwell (sm_100+)",
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
