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

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.qsa import qsa_block_topk_select
from tokenspeed_kernel.ops.attention.qsa.triton import qwen4_exp_qsa_prepare_metadata
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

pytestmark = pytest.mark.skipif(
    not current_platform().is_hopper_plus, reason="CUDA QSA top-k requires NVIDIA SM90+"
)


@pytest.mark.parametrize("enable_pdl", [False, True])
def test_qsa_mtp_padding_topk_cuda_graph(enable_pdl, require):
    require("attention", "qsa_block_topk_select", "cuda", torch.float32, "logits")
    batch, queries, columns, topk = 80, 6, 65600, 512
    seq_lens = torch.full((batch,), 6400, dtype=torch.int32, device="cuda")
    tables = torch.zeros((batch, 1025), dtype=torch.int32, device="cuda")
    torch.manual_seed(35)
    scores = torch.randn((batch * queries, columns), device="cuda")

    def forward():
        metadata = qwen4_exp_qsa_prepare_metadata(
            seq_lens,
            queries,
            batch * queries,
            tables,
            64,
            tables,
            256,
            4,
            enable_pdl=enable_pdl,
        )
        selected = qsa_block_topk_select(
            scores,
            topk,
            lengths=metadata[-1],
            enable_pdl=enable_pdl,
            solution="cuda",
            override=None,
        )
        return selected

    forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = forward()

    # The graph bucket stays fixed as requests finish and its buffers are reused.
    for live_batch in (80, 79, 0, 80):
        seq_lens.fill_(1)
        seq_lens[:live_batch] = 6400
        scores.normal_()
        graph.replay()
        positions = (
            seq_lens.long()[:, None] - queries + torch.arange(queries, device="cuda")
        ).flatten()
        complete = (positions + 1).clamp_min(0).div(4, rounding_mode="floor")
        live_rows = live_batch * queries
        if live_rows:
            valid = (
                torch.arange(1600, device="cuda")[None, :] < complete[:live_rows, None]
            )
            reference = scores[:live_rows, :1600].masked_fill(~valid, -torch.inf)
            live_selected = selected[:live_rows].long()
            assert torch.all(
                (live_selected >= 0) & (live_selected < complete[:live_rows, None])
            )
            ordered = live_selected.sort(dim=1).values
            assert torch.all(ordered[:, 1:] != ordered[:, :-1])
            torch.testing.assert_close(
                scores[:live_rows].gather(1, live_selected).sort(dim=1).values,
                reference.topk(topk, dim=1).values.sort(dim=1).values,
                rtol=0,
                atol=0,
            )
        assert torch.all(selected[live_rows:] == -1)


@pytest.mark.parametrize("topk", [64, 512, 1024, 2048])
def test_qsa_topk_auto_selection_uses_registered_solution(topk, require):
    if topk != 64:
        require("attention", "qsa_block_topk_select", "cuda", torch.float32, "logits")
    scores = torch.arange(4096, device="cuda", dtype=torch.float32)[None, :]
    lengths = torch.tensor([4096], device="cuda", dtype=torch.int32)
    kernel = select_kernel(
        "attention",
        "qsa_block_topk_select",
        format_signature(logits=dense_tensor_format(scores.dtype)),
        traits={"topk": topk},
        solution=None,
        override=None,
    )
    assert kernel.name == (
        "triton_qsa_block_topk_select" if topk == 64 else "cuda_qsa_block_topk_select"
    )
    selected = qsa_block_topk_select(
        scores, topk, lengths=lengths, enable_pdl=False, solution=None, override=None
    )
    torch.testing.assert_close(
        selected.sort().values.long(),
        torch.arange(4096 - topk, 4096, device="cuda")[None, :],
    )


@pytest.mark.parametrize("solution", ["cuda", "triton"])
@pytest.mark.parametrize(("cols", "topk"), [(4096, 512), (65536, 512), (65536, 2048)])
@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("column_stride", [1, 2])
def test_qsa_topk_solutions_preserve_prefixes_ties_and_strides(
    solution, cols, topk, length_dtype, column_stride, require
):
    require("attention", "qsa_block_topk_select", solution, torch.float32, "logits")
    torch.manual_seed(389)
    rows = 7
    scores = torch.empty((rows, cols * column_stride), device="cuda")[
        :, ::column_stride
    ]
    lengths = torch.empty(rows * 2, dtype=length_dtype, device="cuda")[::2]
    lengths.copy_(
        torch.tensor([0, 1, topk - 1, topk, topk + 7, cols, cols], device="cuda")
    )
    columns = torch.arange(cols, device="cuda")
    valid = columns[None, :] < lengths[:, None]
    values = torch.randn(rows, cols, device="cuda").abs()
    values[5].zero_()
    scores.copy_(torch.where(valid, values, float("nan")))
    selected = qsa_block_topk_select(
        scores,
        topk,
        lengths=lengths,
        enable_pdl=False,
        solution=solution,
        override=None,
    )
    reference = scores.masked_fill(~valid, -float("inf"))
    expected = torch.argsort(reference, dim=1, descending=True, stable=True)[:, :topk]
    expected = torch.where(torch.isfinite(reference.gather(1, expected)), expected, -1)
    live = selected >= 0
    torch.testing.assert_close(live.sum(dim=1), lengths.clamp(max=topk).long())
    assert torch.all((selected < lengths[:, None]) | ~live)
    ordered = selected.sort(dim=1).values
    assert not torch.any((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0))
    selected_values = scores.gather(1, selected.clamp(min=0).long())
    selected_values = selected_values.masked_fill(~live, -float("inf"))
    torch.testing.assert_close(
        selected_values.sort(dim=1).values,
        reference.gather(1, expected.clamp(min=0))
        .masked_fill(expected < 0, -float("inf"))
        .sort(dim=1)
        .values,
        rtol=0,
        atol=0,
    )
    if solution == "triton":
        # The portable implementation also guarantees the smaller tied id.
        torch.testing.assert_close(ordered.long(), expected.sort(dim=1).values)


def test_cuda_qsa_topk_quantized_scores_preserve_selected_score_multiset(require):
    require("attention", "qsa_block_topk_select", "cuda", torch.float32, "logits")
    # More than 2048 candidates in the threshold's coarse bin must not discard
    # higher scores before refinement.
    scores = torch.arange(65536, device="cuda").remainder(257).float()[None, :]
    lengths = torch.tensor([scores.shape[1]], dtype=torch.int32, device="cuda")
    selected = qsa_block_topk_select(
        scores, 512, lengths=lengths, enable_pdl=False, solution="cuda", override=None
    )
    torch.testing.assert_close(
        scores.gather(1, selected.long()).sort().values,
        torch.topk(scores, 512).values.sort().values,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("solution", [None, "cuda", "triton"])
@pytest.mark.parametrize(("rows", "cols"), [(0, 0), (0, 16), (3, 0)])
def test_qsa_topk_solutions_handle_empty_inputs(solution, rows, cols, require):
    if solution == "cuda":
        require("attention", "qsa_block_topk_select", "cuda", torch.float32, "logits")
    scores = torch.empty(rows, cols, device="cuda")
    lengths = torch.zeros(rows, device="cuda", dtype=torch.int32)
    selected = qsa_block_topk_select(
        scores, 512, lengths=lengths, enable_pdl=False, solution=solution, override=None
    )
    torch.testing.assert_close(
        selected, torch.full((rows, 512), -1, device="cuda", dtype=torch.int32)
    )


def test_qsa_topk_explicit_cuda_rejects_unsupported_topk(require):
    require("attention", "qsa_block_topk_select", "cuda", torch.float32, "logits")
    scores = torch.zeros(1, 128, device="cuda")
    lengths = torch.tensor([128], device="cuda", dtype=torch.int32)
    with pytest.raises(NoKernelFoundError):
        qsa_block_topk_select(
            scores,
            64,
            lengths=lengths,
            enable_pdl=False,
            solution="cuda",
            override=None,
        )
