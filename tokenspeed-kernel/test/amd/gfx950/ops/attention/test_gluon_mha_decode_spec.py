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

"""Paged grouped-query block verification and flattened non-causal drafts.

Long histories share physical cache pages across requests to keep the isolated
operator fixture small; distinct/reordered page tables still exercise address
translation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch
from tokenspeed_kernel.ops.attention.mha import mha_decode_with_kvcache
from utils import is_cdna4

MAX_CONTEXT = 50176


@dataclass(frozen=True)
class DecodeCase:
    concurrency: int
    mode: str
    context: int

    @property
    def width(self):
        return 1 if self.mode == "draft" else 16

    @property
    def heads(self):
        return 5 if self.mode == "draft" else 6

    @property
    def window(self):
        return {"full": -1, "window": 511, "draft": 4095}[self.mode]


def make_inputs(case: DecodeCase):
    """Make strided queries like the runtime's projection view and page64 KV."""
    torch.manual_seed(417)
    width = 16
    rows = case.concurrency * width
    pages = (case.context + 63) // 64
    table_pages = (MAX_CONTEXT + 63) // 64
    # Leave a gap after Q, as in a view into a packed projection result.
    q = torch.randn(rows, case.heads + 2, 128, dtype=torch.bfloat16, device="cuda")[
        :, : case.heads
    ]
    k = torch.randn(pages + 3, 64, 1, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    table = torch.stack(
        [
            (torch.arange(table_pages, device="cuda", dtype=torch.int32) + i + 1)
            % (pages + 3)
            for i in range(case.concurrency)
        ]
    )
    lengths = torch.tensor(
        [max(width + 1, case.context - (i % 4)) for i in range(case.concurrency)],
        device="cuda",
        dtype=torch.int32,
    )
    if case.mode == "draft":
        # All 16 block positions see the same block-end length: non-causal
        # attention expressed through the shared one-query metadata contract.
        table = table.repeat_interleave(16, dim=0)
        lengths = lengths.repeat_interleave(16)
    return {
        "q": q,
        "k_cache": k,
        "v_cache": v,
        "page_table": table,
        "cache_seqlens": lengths,
        "max_seqlen_k": MAX_CONTEXT,
        "max_seqlen_q": case.width,
        "window_left": case.window,
    }


@contextmanager
def reference_precision():
    """Require full FP32 matmuls without changing later tests' precision."""
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@reference_precision()
def reference(inputs):
    """Independent FP32 dense oracle; only one request is materialized at once."""
    q = inputs["q"].float()
    result = torch.empty_like(q)
    width = inputs["max_seqlen_q"]
    lengths = inputs["cache_seqlens"].cpu().tolist()
    for batch, length in enumerate(lengths):
        blocks = inputs["page_table"][batch, : (length + 63) // 64].long()
        k = inputs["k_cache"][blocks].reshape(-1, 128)[:length].float()
        v = inputs["v_cache"][blocks].reshape(-1, 128)[:length].float()
        query = q[batch * width : (batch + 1) * width]
        logits = torch.einsum("qhd,kd->qhk", query, k) * (128**-0.5)
        ends = length - width + 1 + torch.arange(width, device=q.device)
        positions = torch.arange(length, device=q.device)
        mask = positions[None, :] < ends[:, None]
        if inputs["window_left"] >= 0:
            mask &= positions[None, :] >= (ends - inputs["window_left"] - 1)[:, None]
        probabilities = torch.softmax(
            logits.masked_fill(~mask[:, None, :], -torch.inf), dim=-1
        )
        result[batch * width : (batch + 1) * width] = torch.einsum(
            "qhk,kd->qhd", probabilities, v
        )
    return result


def changed_inputs(inputs):
    """Change values only, preserving every captured address and host bound."""
    inputs["q"].mul_(-0.75)
    inputs["cache_seqlens"].sub_(1)
    table = inputs["page_table"]
    table.copy_(table.roll(1, dims=1))


def check_output(actual, expected):
    torch.testing.assert_close(actual.float(), expected, atol=0.006, rtol=0.006)


@pytest.mark.parametrize("concurrency", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("mode", ["full", "window", "draft"])
@pytest.mark.parametrize("context", [17, 63, 64, 65, 8192, 50000])
def test_spec_decode_graph_replay(concurrency, mode, context):
    if not is_cdna4():
        pytest.skip("requires gfx950")
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.decode import (
        _mha_decode_sliding_split_atomic,
        gluon_mha_decode_gfx950,
        gluon_mha_decode_reduce_gfx950,
        gluon_mha_decode_sliding_gfx950,
    )
    from utils import assert_no_triton_compile

    # Measure this case, excluding peak allocations from earlier tests.
    torch.cuda.reset_peak_memory_stats()
    case = DecodeCase(concurrency, mode, context)
    inputs = make_inputs(case)
    expected = reference(inputs)
    for solution in ("triton", "gluon"):
        check_output(
            mha_decode_with_kvcache(
                **inputs, solution=solution, block_draft=False, batch_invariant=False
            ),
            expected,
        )
    # A draft's block positions share their request's table row and length:
    # read as uniform groups, they take each request's keys once, bit for bit.
    groups = 16 if mode == "draft" else 1
    if groups > 1:
        assert_uniform_groups_bitwise(inputs, groups)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = mha_decode_with_kvcache(
            **inputs,
            solution="gluon",
            block_draft=False,
            batch_invariant=False,
            uniform_row_groups=groups,
        )
    graph.replay()
    check_output(actual, expected)
    changed_inputs(inputs)
    expected = reference(inputs)
    with assert_no_triton_compile(
        gluon_mha_decode_gfx950,
        gluon_mha_decode_reduce_gfx950,
        gluon_mha_decode_sliding_gfx950,
        _mha_decode_sliding_split_atomic,
    ):
        # Eager dispatch under the guard verifies dynamic values do not become
        # specialization keys; graph replay alone could not establish that.
        check_output(
            mha_decode_with_kvcache(
                **inputs, solution="gluon", block_draft=False, batch_invariant=False
            ),
            expected,
        )
        if groups > 1:
            assert_uniform_groups_bitwise(inputs, groups)
        graph.replay()
        check_output(actual, expected)
    assert torch.cuda.max_memory_allocated() < 8 * 1024**3


def assert_uniform_groups_bitwise(inputs, groups):
    """Uniform row groups give the one-row requests' bits: on the kernel that
    reads them (gluon) and on one that serves the rows as they are (triton)."""
    for solution in ("gluon", "triton"):
        rows = mha_decode_with_kvcache(
            **inputs, solution=solution, block_draft=False, batch_invariant=False
        )
        grouped = mha_decode_with_kvcache(
            **inputs,
            solution=solution,
            block_draft=False,
            batch_invariant=False,
            uniform_row_groups=groups,
        )
        assert torch.equal(grouped.view(torch.int16), rows.view(torch.int16)), solution


@pytest.mark.parametrize("running", [8, 12, 16])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("window", [2047, -1])
def test_uniform_row_groups_keep_every_bit_at_draft_shapes(running, kv_dtype, window):
    """A block drafter's attention (8 positions, 16 query / 2 KV heads, dim 128)
    over ragged lengths: tiny, around a page boundary, around the window edge
    and long."""
    if not is_cdna4():
        pytest.skip("requires gfx950")
    torch.manual_seed(running)
    positions, page = 8, 64
    lengths = torch.tensor(
        [
            [1, 9, 63, 64, 65, 2047, 2048, 2049, 30001, 70000, 88000, 129][i % 12]
            for i in range(running)
        ],
        device="cuda",
        dtype=torch.int32,
    )
    width = int(lengths.max()) // page + 2
    pages = running * width
    k = torch.randn(pages, page, 2, 128, device="cuda").to(kv_dtype)
    v = torch.randn(pages, page, 2, 128, device="cuda").to(kv_dtype)
    table = torch.randperm(pages, device="cuda").view(running, width).to(torch.int32)
    inputs = {
        "q": torch.randn(running * positions, 16, 128, device="cuda").to(kv_dtype),
        "k_cache": k,
        "v_cache": v,
        "page_table": table.repeat_interleave(positions, dim=0),
        "cache_seqlens": lengths.repeat_interleave(positions),
        "max_seqlen_k": 262144,
        "max_seqlen_q": 1,
        "window_left": window,
    }
    rows = mha_decode_with_kvcache(
        **inputs, solution="gluon", block_draft=False, batch_invariant=False
    )
    grouped = mha_decode_with_kvcache(
        **inputs,
        solution="gluon",
        block_draft=False,
        batch_invariant=False,
        uniform_row_groups=positions,
    )
    assert torch.equal(grouped.view(torch.int16), rows.view(torch.int16))


@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("window", [2047, -1])
def test_uniform_row_groups_with_an_output_gate_keep_every_bit(kv_dtype, window):
    """A grouped block with an output gate (64-wide heads, the gate's range)
    stores what its one-row requests followed by the separate gated norm
    give, bit for bit, without launching that op."""
    if not is_cdna4():
        pytest.skip("requires gfx950")
    from tokenspeed_kernel.ops.layernorm import weightless_rmsnorm_gated_sigmoid
    from torch.profiler import ProfilerActivity, profile

    torch.manual_seed(11)
    running, positions, page, heads, dim = 9, 8, 64, 16, 64
    lengths = torch.tensor(
        [1, 63, 64, 65, 2047, 2048, 2049, 30001, 129], device="cuda", dtype=torch.int32
    )
    width = int(lengths.max()) // page + 2
    pages = running * width
    k = torch.randn(pages, page, 2, dim, device="cuda").to(kv_dtype)
    v = torch.randn(pages, page, 2, dim, device="cuda").to(kv_dtype)
    table = torch.randperm(pages, device="cuda").view(running, width).to(torch.int32)
    rows = running * positions
    inputs = {
        "q": torch.randn(rows, heads, dim, device="cuda").to(kv_dtype),
        "k_cache": k,
        "v_cache": v,
        "page_table": table.repeat_interleave(positions, dim=0),
        "cache_seqlens": lengths.repeat_interleave(positions),
        "max_seqlen_k": 262144,
        "max_seqlen_q": 1,
        "window_left": window,
    }
    gate = torch.randn(rows, heads * dim, device="cuda").bfloat16()
    plain = mha_decode_with_kvcache(
        **inputs, solution="gluon", block_draft=False, batch_invariant=False
    )
    expected = weightless_rmsnorm_gated_sigmoid(
        plain.view(rows, heads * dim), gate, heads, dim, 1e-6, False
    )
    # A fallback to the separate op would give the same bits; the profile
    # shows the kernel's own store applied the gate.
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        grouped = mha_decode_with_kvcache(
            **inputs,
            solution="gluon",
            block_draft=False,
            batch_invariant=False,
            uniform_row_groups=positions,
            output_gate=gate,
            output_gate_eps=1e-6,
        )
        torch.cuda.synchronize()
    assert not any("gated_sigmoid" in event.name for event in prof.events())
    assert torch.equal(
        grouped.view(rows, -1).view(torch.int16), expected.view(torch.int16)
    )
