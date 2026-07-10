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

"""Tests for the CuTe DSL (cluster) DSA decode top-k ``cute_dsl_decode_topk``.

Covers causal-window ``torch.topk`` parity across batch/kv-len/top-k/next_n
(including ``window < top_k`` and speculative ``next_n > 1``), in-window indices,
in-place ``out`` reuse, drop-in equivalence with the ``deterministic_decode_topk``
path via ``local_topk_to_global_slots``, and CUDA-graph capture/replay. NVIDIA
Blackwell (sm_100+) only; skips elsewhere.
"""

from __future__ import annotations

import pytest
import torch

dsa_cute = pytest.importorskip("tokenspeed_kernel.ops.attention.cute_dsl.dsa_topk")
cute_dsl_decode_topk = dsa_cute.cute_dsl_decode_topk
has_cute_dsl_decode_topk = dsa_cute.has_cute_dsl_decode_topk

requires_kernel = pytest.mark.skipif(
    not (torch.cuda.is_available() and has_cute_dsl_decode_topk()),
    reason="CuTe DSL DSA decode top-k requires NVIDIA Blackwell (sm_100+)",
)

# (batch, kv_len, top_k, next_n)
_GRID = [
    (1, 8192, 2048, 1),
    (4, 8192, 2048, 1),
    (8, 4096, 2048, 1),
    (16, 2048, 512, 1),
    (2, 65536, 2048, 1),
    (3, 4096, 2048, 2),  # speculative decode next_n=2
    (2, 6000, 1024, 3),  # speculative decode next_n=3
]


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
@pytest.mark.parametrize("bs,kv,topk,next_n", _GRID)
def test_parity_with_causal_window_reference(bs, kv, topk, next_n):
    torch.manual_seed(0)
    num_rows = bs * next_n
    logits = torch.randn(num_rows, kv, device="cuda", dtype=torch.float32)
    # Per-request lengths straddle top_k so both the window<top_k and
    # window>=top_k regimes are exercised.
    seq_lens = torch.randint(
        max(1, topk // 4), kv + 1, (bs,), device="cuda", dtype=torch.int32
    )
    out = torch.empty(num_rows, topk, device="cuda", dtype=torch.int32)
    ret = cute_dsl_decode_topk(logits, seq_lens, topk, next_n=next_n, out=out)

    assert ret.data_ptr() == out.data_ptr(), "out must be written in place"
    assert ret.dtype == torch.int32 and ret.shape == (num_rows, topk)

    got = _gathered_topk_values(logits, ret, seq_lens, topk, next_n)
    ref = _reference_topk_values(logits, seq_lens, topk, next_n)
    assert torch.equal(got, ref), "selected top-k values differ from reference"


@requires_kernel
def test_window_shorter_than_topk_early_decode():
    """Early-decode regime: every row's causal window is far below top_k."""
    torch.manual_seed(1)
    bs, kv, topk = 8, 8192, 2048
    logits = torch.randn(bs, kv, device="cuda", dtype=torch.float32)
    seq_lens = torch.randint(16, 400, (bs,), device="cuda", dtype=torch.int32)
    out = torch.empty(bs, topk, device="cuda", dtype=torch.int32)
    cute_dsl_decode_topk(logits, seq_lens, topk, next_n=1, out=out)
    got = _gathered_topk_values(logits, out, seq_lens, topk, 1)
    ref = _reference_topk_values(logits, seq_lens, topk, 1)
    assert torch.equal(got, ref)


@requires_kernel
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dtype_coverage(dtype):
    torch.manual_seed(2)
    bs, kv, topk = 4, 8192, 2048
    logits = torch.randn(bs, kv, device="cuda", dtype=dtype)
    seq_lens = torch.full((bs,), kv, device="cuda", dtype=torch.int32)
    out = torch.empty(bs, topk, device="cuda", dtype=torch.int32)
    cute_dsl_decode_topk(logits, seq_lens, topk, next_n=1, out=out)
    # Compare against topk over the same (low-precision) logits.
    got = _gathered_topk_values(logits.float(), out, seq_lens, topk, 1)
    ref = _reference_topk_values(logits.float(), seq_lens, topk, 1)
    assert torch.equal(got, ref)


@requires_kernel
def test_allocates_output_when_out_is_none():
    torch.manual_seed(3)
    bs, kv, topk = 4, 4096, 2048
    logits = torch.randn(bs, kv, device="cuda", dtype=torch.float32)
    seq_lens = torch.full((bs,), kv, device="cuda", dtype=torch.int32)
    ret = cute_dsl_decode_topk(logits, seq_lens, topk, next_n=1)
    assert ret.dtype == torch.int32 and ret.shape == (bs, topk)
    got = _gathered_topk_values(logits, ret, seq_lens, topk, 1)
    ref = _reference_topk_values(logits, seq_lens, topk, 1)
    assert torch.equal(got, ref)


@requires_kernel
@pytest.mark.parametrize(
    "bs,max_pages,topk,next_n",
    [
        (4, 40, 2048, 1),  # windows straddle top_k
        (3, 48, 2048, 2),  # speculative decode
        (8, 16, 512, 1),  # tiny windows, small top_k
    ],
)
def test_dropin_equivalence_with_persistent_radix(bs, max_pages, topk, next_n):
    """cute_dsl + slot mapping == deterministic_decode_topk + slot mapping.

    This is the integration contract: composed with
    ``local_topk_to_global_slots`` the CuTe DSL path must produce identical
    global KV slots and valid counts as the persistent-radix path it replaces.
    """
    flashinfer_topk = pytest.importorskip(
        "tokenspeed_kernel.ops.attention.flashinfer.dsa_topk"
    )
    if not flashinfer_topk.has_ragged_decode_topk():
        pytest.skip("ragged persistent_topk CUDA kernel unavailable")
    from tokenspeed_kernel.ops.attention.triton.dsa_topk import (
        local_topk_to_global_slots,
    )

    page_size = 64
    num_cols = max_pages * page_size
    num_rows = bs * next_n
    torch.manual_seed(7)
    logits = torch.randn(num_rows, num_cols, device="cuda", dtype=torch.float32)
    seq_lens = torch.randint(16, num_cols + 1, (bs,), device="cuda", dtype=torch.int32)
    block_table = (
        torch.arange(bs * max_pages, device="cuda", dtype=torch.int32).view(
            bs, max_pages
        )
        + 1
    )

    def _map(local_offsets):
        slots = torch.empty(num_rows, topk, device="cuda", dtype=torch.int32)
        lens = torch.empty(num_rows, device="cuda", dtype=torch.int32)
        local_topk_to_global_slots(
            local_topk_offsets=local_offsets,
            block_table=block_table,
            block_size=page_size,
            seq_lens=seq_lens,
            q_len_per_req=next_n,
            out=slots,
            lens_out=lens,
        )
        return slots, lens

    loc_cute = torch.empty(num_rows, topk, device="cuda", dtype=torch.int32)
    cute_dsl_decode_topk(logits, seq_lens, topk, next_n=next_n, out=loc_cute)
    slots_cute, lens_cute = _map(loc_cute)

    loc_ref = torch.empty(num_rows, topk, device="cuda", dtype=torch.int32)
    flashinfer_topk.deterministic_decode_topk(
        logits,
        loc_ref,
        topk,
        lengths=seq_lens,
        q_len_per_req=next_n,
        workspace=torch.empty((1 << 20,), dtype=torch.uint8, device="cuda"),
        max_seq_len=num_cols,
    )
    slots_ref, lens_ref = _map(loc_ref)

    assert torch.equal(lens_cute, lens_ref), "valid counts differ"
    for r in range(num_rows):
        n = int(lens_ref[r])
        a = torch.sort(slots_cute[r, :n]).values
        b = torch.sort(slots_ref[r, :n]).values
        assert torch.equal(a, b), f"row {r}: selected KV slot set differs"


@requires_kernel
def test_cuda_graph_capture_replay():
    """The decode path runs under CUDA graphs; capture/replay must be correct."""
    torch.manual_seed(4)
    bs, kv, topk, next_n = 4, 8192, 2048, 1
    logits = torch.randn(bs, kv, device="cuda", dtype=torch.float32)
    seq_lens = torch.randint(topk // 2, kv + 1, (bs,), device="cuda", dtype=torch.int32)
    out = torch.empty(bs, topk, device="cuda", dtype=torch.int32)

    # Warm up (JIT compile + first scratch allocation happen here, eagerly).
    cute_dsl_decode_topk(logits, seq_lens, topk, next_n=next_n, out=out)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cute_dsl_decode_topk(logits, seq_lens, topk, next_n=next_n, out=out)

    # New inputs, replay, and confirm the captured graph recomputes correctly.
    new_logits = torch.randn(bs, kv, device="cuda", dtype=torch.float32)
    new_seq_lens = torch.randint(
        topk // 2, kv + 1, (bs,), device="cuda", dtype=torch.int32
    )
    logits.copy_(new_logits)
    seq_lens.copy_(new_seq_lens)
    graph.replay()
    torch.cuda.synchronize()

    got = _gathered_topk_values(logits, out, seq_lens, topk, next_n)
    ref = _reference_topk_values(logits, seq_lens, topk, next_n)
    assert torch.equal(got, ref), "graph replay produced wrong top-k"
