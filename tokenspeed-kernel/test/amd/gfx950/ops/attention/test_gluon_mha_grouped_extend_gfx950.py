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

"""Non-power-of-two GQA groups, speculative rows and live paged metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch
from utils import is_amd


@pytest.fixture
def require_gfx950():
    from utils import is_cdna4

    if not is_cdna4():
        pytest.skip("AMD gfx950 is required for Gluon grouped MHA device tests")


@pytest.mark.parametrize("group", [1, 2, 4, 5, 6, 8, 12])
@pytest.mark.parametrize("query_rows", [1, 16, 64, 848, 896, 8192])
def test_select_extend_tile(group, query_rows):
    # Tile selection is CPU-only; the AMD kernel package is optional on other hosts.
    if not is_amd():
        pytest.importorskip(
            "tokenspeed_kernel_amd", reason="AMD kernel package is optional"
        )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.extend import (
        _select_extend_tile,
    )

    block_m, block_q, block_n, warps = _select_extend_tile(query_rows, group)
    padded_group = 1 << (group - 1).bit_length()
    assert block_m & (block_m - 1) == 0
    assert block_q & (block_q - 1) == 0
    assert block_m == block_q * padded_group
    assert block_n == 64 and warps in (2, 4)
    # The tile must cover each real (position, head) exactly once.
    actual = [
        (row // padded_group, row % padded_group)
        for row in range(block_m)
        if row % padded_group < group
    ]
    assert actual == [(pos, head) for pos in range(block_q) for head in range(group)]


@dataclass
class PagedCase:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    table: torch.Tensor
    lengths: torch.Tensor
    cu_q: torch.Tensor
    cu_kv: torch.Tensor
    queries: list[int]
    max_q: int
    max_k: int
    window: int
    causal: bool
    sinks: torch.Tensor | None

    def invoke(self, solution: str, *, return_lse: bool):
        from tokenspeed_kernel.ops.attention.mha import mha_extend_with_kvcache

        return mha_extend_with_kvcache(
            q=self.q,
            cu_seqlens_q=self.cu_q,
            cu_seqlens_kv=self.cu_kv,
            k_cache=self.k,
            v_cache=self.v,
            page_table=self.table,
            cache_seqlens=self.lengths,
            is_causal=self.causal,
            window_left=self.window,
            sinks=self.sinks,
            return_lse=return_lse,
            max_seqlen_q=self.max_q,
            max_seqlen_k=self.max_k,
            solution=solution,
        )


def make_case(queries, prefixes, *, heads, kv_heads, window, causal, sinks):
    torch.manual_seed(1357)
    lengths = [q + p for q, p in zip(queries, prefixes, strict=True)]
    pages_per_request = math.ceil(max(lengths) / 64)
    page_count = len(queries) * pages_per_request + 1
    q = torch.randn((sum(queries), heads, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        (page_count, 64, kv_heads, 128), device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    table = torch.arange(1, page_count, device="cuda", dtype=torch.int32).view(
        len(queries), pages_per_request
    )
    # Noncontiguous physical pages catch accidental use of logical token order.
    table = table.flip(1).contiguous()
    cu_q = [0]
    cu_kv = [0]
    for query, length in zip(queries, lengths, strict=True):
        cu_q.append(cu_q[-1] + query)
        cu_kv.append(cu_kv[-1] + length)
    return PagedCase(
        q,
        k,
        v,
        table,
        torch.tensor(lengths, device="cuda", dtype=torch.int32),
        torch.tensor(cu_q, device="cuda", dtype=torch.int32),
        torch.tensor(cu_kv, device="cuda", dtype=torch.int32),
        list(queries),
        max(queries),
        max(lengths),
        window,
        causal,
        torch.randn(heads, device="cuda", dtype=torch.bfloat16) if sinks else None,
    )


def reference(case: PagedCase):
    """FP32 dense oracle, chunked only along independent query rows."""
    outputs, lses = [], []
    start = 0
    group = case.q.shape[1] // case.k.shape[2]
    for request, query_rows in enumerate(case.queries):
        length = int(case.lengths[request].item())
        pages = case.table[request].long()
        k = case.k[pages].flatten(0, 1)[:length].float().repeat_interleave(group, 1)
        v = case.v[pages].flatten(0, 1)[:length].float().repeat_interleave(group, 1)
        key_pos = torch.arange(length, device="cuda")
        prefix = length - query_rows
        for offset in range(0, query_rows, 32):
            end = min(offset + 32, query_rows)
            q = case.q[start + offset : start + end].float()
            scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(128)
            positions = prefix + torch.arange(offset, end, device="cuda")
            mask = torch.ones((end - offset, length), device="cuda", dtype=torch.bool)
            if case.causal:
                mask &= key_pos[None, :] <= positions[:, None]
            if case.window >= 0:
                mask &= key_pos[None, :] >= positions[:, None] - case.window
            scores.masked_fill_(~mask[None], -float("inf"))
            if case.sinks is not None:
                sink_scores = case.sinks.float()[:, None, None].expand(
                    -1, end - offset, 1
                )
                scores = torch.cat((scores, sink_scores), dim=-1)
            lse = torch.logsumexp(scores, dim=-1)
            p = torch.softmax(scores, dim=-1)[..., :length]
            outputs.append(torch.einsum("hqk,khd->qhd", p, v))
            lses.append(lse.transpose(0, 1))
        start += query_rows
    return torch.cat(outputs), torch.cat(lses)


def assert_reference(case: PagedCase):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        expected, expected_lse = reference(case)
        actual, actual_lse = case.invoke("gluon", return_lse=True)
        ordinary = case.invoke("gluon", return_lse=False)
        # The public MHA LSE tests use these existing output/LSE tolerances.
        torch.testing.assert_close(actual.float(), expected, rtol=8e-2, atol=8e-2)
        torch.testing.assert_close(ordinary.float(), expected, rtol=8e-2, atol=8e-2)
        torch.testing.assert_close(actual_lse, expected_lse, rtol=8e-2, atol=8e-2)
        assert torch.isfinite(actual).all() and torch.isfinite(actual_lse).all()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("window", [-1, 511])
@pytest.mark.usefixtures("require_gfx950")
def test_speculative_verify_rows(batch, window):
    case = make_case(
        [16] * batch,
        [50000 - 7 * i for i in range(batch)],
        heads=6,
        kv_heads=1,
        window=window,
        causal=True,
        sinks=False,
    )
    assert_reference(case)


@pytest.mark.parametrize("query_rows", [848, 896, 8192])
@pytest.mark.parametrize("window", [-1, 511])
@pytest.mark.usefixtures("require_gfx950")
def test_cached_prefill_chunks(query_rows, window):
    case = make_case(
        [query_rows],
        [50000 - query_rows],
        heads=6,
        kv_heads=1,
        window=window,
        causal=True,
        sinks=False,
    )
    assert_reference(case)


@pytest.mark.parametrize("window", [-1, 511])
@pytest.mark.usefixtures("require_gfx950")
def test_ragged_mixed_prefill_and_verify(window):
    case = make_case(
        [848, 16, 16],
        [49152, 49997, 50019],
        heads=6,
        kv_heads=1,
        window=window,
        causal=True,
        sinks=True,
    )
    assert_reference(case)


@pytest.mark.parametrize(
    "heads,kv_heads,causal,window", [(5, 1, False, 4095), (12, 2, True, 511)]
)
@pytest.mark.usefixtures("require_gfx950")
def test_draft_and_multiple_kv_head_groups(heads, kv_heads, causal, window):
    case = make_case(
        [16, 5],
        [127, 211],
        heads=heads,
        kv_heads=kv_heads,
        window=window,
        causal=causal,
        sinks=True,
    )
    assert_reference(case)


@pytest.mark.parametrize("window", [-1, 511])
@pytest.mark.usefixtures("require_gfx950")
def test_changed_metadata_graph_replay_without_recompile(window):
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.extend import (
        gluon_mha_extend_gfx950,
        gluon_mha_extend_reduce_gfx950,
        gluon_mha_extend_split_gfx950,
    )
    from utils import assert_no_triton_compile

    case = make_case(
        [16, 16],
        [1001, 977],
        heads=6,
        kv_heads=1,
        window=window,
        causal=True,
        sinks=True,
    )
    case.max_q = 32
    case.max_k = 2048
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            case.invoke("gluon", return_lse=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output, lse = case.invoke("gluon", return_lse=True)
    torch.cuda.synchronize()
    for queries, lengths in [([9, 23], [1021, 997]), ([21, 11], [1007, 1013])]:
        case.queries = queries
        case.cu_q.copy_(
            torch.tensor([0, queries[0], 32], device="cuda", dtype=torch.int32)
        )
        case.lengths.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        case.cu_kv.copy_(
            torch.tensor(
                [0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32
            )
        )
        case.table.copy_(case.table.flip(1))
        with assert_no_triton_compile(
            gluon_mha_extend_gfx950,
            gluon_mha_extend_split_gfx950,
            gluon_mha_extend_reduce_gfx950,
        ):
            graph.replay()
            torch.cuda.synchronize()
            eager, eager_lse = case.invoke("gluon", return_lse=True)
        torch.testing.assert_close(output, eager, rtol=0, atol=0)
        torch.testing.assert_close(lse, eager_lse, rtol=0, atol=0)
        assert_reference(case)
    graph.reset()


@pytest.mark.parametrize("query_rows", [848, 896, 8192])
@pytest.mark.parametrize("window", [-1, 511])
@pytest.mark.usefixtures("require_gfx950")
def test_initial_prefill_groups(query_rows, window):
    from tokenspeed_kernel.ops.attention.mha import mha_prefill

    case = make_case(
        [query_rows], [0], heads=6, kv_heads=1, window=window, causal=True, sinks=False
    )
    k = case.k[case.table[0].long()].flatten(0, 1)[:query_rows]
    v = case.v[case.table[0].long()].flatten(0, 1)[:query_rows]
    actual = mha_prefill(
        q=case.q,
        k=k,
        v=v,
        cu_seqlens=case.cu_q,
        cu_seqlens_cpu=[0, query_rows],
        max_seqlen=query_rows,
        window_left=window,
        solution="gluon",
    )
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        expected, _ = reference(case)
        torch.testing.assert_close(actual.float(), expected, rtol=8e-2, atol=8e-2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
