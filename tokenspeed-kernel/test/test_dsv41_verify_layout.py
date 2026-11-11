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

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsv41 import verify_layout


def reference_layout(
    scores,
    query_capacity,
    rows_out,
    indptr_out,
    counts_out,
    valid_out,
    latest_scores=None,
    request_slots=None,
    active_count=None,
):
    if latest_scores is not None:
        latest = latest_scores.index_select(0, request_slots)
        scores = torch.where(
            torch.isfinite(latest).all(dim=1, keepdim=True), latest, scores
        )
    batch_size, draft_width = scores.shape
    width = draft_width + 1
    live = torch.arange(batch_size, device=scores.device) < (
        batch_size if active_count is None else int(active_count.item())
    )
    finite = torch.isfinite(scores).all(dim=1)
    survival = torch.sigmoid(scores.clamp(-60, 60)).cumprod(dim=1)
    gains = torch.where(
        finite[:, None], survival, torch.full_like(survival, float("inf"))
    ).flatten()
    # Stable ties preserve a prefix within each request, including underflowed
    # probabilities and unknown-score rows requiring their complete draft.
    gains = torch.where(
        live[:, None].expand(-1, draft_width).flatten(), gains, -torch.inf
    )
    order = torch.argsort(gains, descending=True, stable=True)
    extra = torch.zeros_like(gains, dtype=torch.int64)
    extra.scatter_(0, order[: query_capacity - batch_size], 1)
    extra *= live[:, None].expand(-1, draft_width).flatten()
    counts = 1 + extra.reshape(batch_size, draft_width).sum(dim=1)
    valid = ((~finite) & live).sum() * draft_width <= query_capacity - batch_size
    counts = torch.where(valid, counts, torch.ones_like(counts))
    indptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    positions = torch.arange(query_capacity, device=scores.device, dtype=torch.int64)
    owners = torch.searchsorted(indptr[1:], positions, right=True).clamp_max(
        batch_size - 1
    )
    rows = owners * width + positions - indptr[owners]
    rows_out.copy_(torch.where(positions < indptr[-1], rows, torch.zeros_like(rows)))
    indptr_out.copy_(indptr)
    counts_out.copy_(counts)
    valid_out.copy_(valid)


def outputs(scores, capacity):
    bs = scores.shape[0]
    return dict(
        rows_out=torch.empty(capacity, dtype=torch.int64, device=scores.device),
        indptr_out=torch.empty(bs + 1, dtype=torch.int64, device=scores.device),
        counts_out=torch.empty(bs, dtype=torch.int64, device=scores.device),
        valid_out=torch.empty((), dtype=torch.bool, device=scores.device),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "bs,draft_width,extra", [(1, 5, 0), (4, 3, 5), (32, 5, 120), (128, 5, 640)]
)
def test_layout_matches_stable_prefix_reference(bs, draft_width, extra):
    scores = torch.randn(
        bs, draft_width, generator=torch.Generator().manual_seed(42)
    ).to("cuda")
    capacity = bs + extra
    out = outputs(scores, capacity)
    verify_layout(scores, query_capacity=capacity, **out)
    probs = torch.sigmoid(scores.cpu().clamp(-60, 60)).cumprod(1).tolist()
    ordered = sorted(
        ((p, r, d) for r, row in enumerate(probs) for d, p in enumerate(row)),
        key=lambda x: (-x[0], x[1], x[2]),
    )
    counts = [1] * bs
    for _, row, _ in ordered[:extra]:
        counts[row] += 1
    rows = [
        r * (draft_width + 1) + d
        for r, count in enumerate(counts)
        for d in range(count)
    ]
    assert out["valid_out"].item()
    assert out["counts_out"].tolist() == counts
    assert out["rows_out"].tolist() == rows
    assert out["indptr_out"][-1].item() == capacity


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_unknown_scores_require_full_prefix_and_ties_are_stable():
    scores = torch.tensor([[float("nan"), 0.0], [0.0, 0.0]], device="cuda")
    out = outputs(scores, 4)
    verify_layout(scores, query_capacity=4, **out)
    assert out["valid_out"].item()
    assert out["counts_out"].tolist() == [3, 1]
    out = outputs(scores, 3)
    verify_layout(scores, query_capacity=3, **out)
    assert not out["valid_out"].item()
    assert out["counts_out"].tolist() == [1, 1]
    assert out["rows_out"].tolist() == [0, 3, 0]
    scores.fill_(-1000)
    out = outputs(scores, 4)
    verify_layout(scores, query_capacity=4, **out)
    assert out["counts_out"].tolist() == [2, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_graph_replay_reads_changed_scores_and_keeps_output_storage():
    scores = torch.zeros((4, 5), device="cuda")
    out = outputs(scores, 12)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            verify_layout(scores, query_capacity=12, **out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        verify_layout(scores, query_capacity=12, **out)
    for row in range(4):
        scores.fill_(-10)
        scores[row].fill_(10)
        graph.replay()
        expected = outputs(scores, 12)
        verify_layout(scores, query_capacity=12, **expected)
        for name, value in out.items():
            torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
    graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("bs", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 682])
def test_layout_matches_torch_reference(bs):
    for seed in range(8):
        scores = torch.randn(
            bs, 5, generator=torch.Generator().manual_seed(seed)
        ).cuda()
        if seed == 1:
            scores.zero_()
        elif seed == 2:
            scores.fill_(-1000)
        elif seed == 3:
            scores[::2, 0] = float("nan")
        elif seed == 4:
            scores[::2, 1] = float("inf")
        elif seed == 5:
            scores.fill_(1000)
        elif seed == 6:
            scores[:, 1] = -60
        for capacity in sorted({bs, bs + 1, bs * 3, bs * 6 - 1, bs * 6}):
            expected = outputs(scores, capacity)
            actual = outputs(scores, capacity)
            reference_layout(scores, capacity, **expected)
            verify_layout(scores, query_capacity=capacity, **actual)
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_triton_layout_runtime_capacities_do_not_compile():
    from tokenspeed_kernel.ops.attention.dsv41.triton import (
        _verify_layout_kernel,
        triton_dsv41_verify_layout,
    )
    from utils import assert_no_triton_compile

    scores = torch.randn(32, 5, device="cuda")
    for capacity in [32, 128]:
        triton_dsv41_verify_layout(scores, capacity, **outputs(scores, capacity))
    with assert_no_triton_compile(_verify_layout_kernel):
        for bs, capacity in [(32, 48), (32, 176), (24, 72), (25, 150)]:
            current = scores[:bs]
            out = outputs(current, capacity)
            triton_dsv41_verify_layout(current, capacity, **out)
            assert out["indptr_out"][-1].item() == capacity


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fused_latest_scores_replay_and_history_fallback():

    history = torch.randn(8, 5, device="cuda")
    latest = torch.randn(32, 5, device="cuda")
    slots = torch.arange(8, dtype=torch.int64, device="cuda")
    out = outputs(history, 24)

    def run():
        verify_layout(
            history, query_capacity=24, latest_scores=latest, request_slots=slots, **out
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    for seed in range(8):
        latest.normal_()
        latest[::3, seed % 5] = float("nan")
        history.normal_()
        slots.copy_(torch.randperm(32, device="cuda")[:8])
        graph.replay()
        expected = outputs(history, 24)
        reference_layout(
            history, 24, **expected, latest_scores=latest, request_slots=slots
        )
        for name in expected:
            torch.testing.assert_close(out[name], expected[name], atol=0, rtol=0)
    graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_latest_layout_runtime_shapes_do_not_compile():
    from tokenspeed_kernel.ops.attention.dsv41.triton import _verify_layout_kernel
    from utils import assert_no_triton_compile

    history = torch.randn(32, 5, device="cuda")
    latest = torch.randn(128, 5, device="cuda")
    slots = torch.arange(32, device="cuda")

    def run(bs, capacity, table_rows):
        out = outputs(history[:bs], capacity)
        verify_layout(
            history[:bs],
            query_capacity=capacity,
            latest_scores=latest[:table_rows],
            request_slots=slots[:bs],
            **out,
        )
        assert out["indptr_out"][-1].item() == capacity

    run(32, 128, 128)
    with assert_no_triton_compile(_verify_layout_kernel):
        run(24, 72, 64)
        run(25, 100, 96)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("case", ["scores_stride", "latest_stride", "capacity"])
def test_unsupported_layout_rejected(case):
    bs = 683 if case == "capacity" else 4
    scores = torch.randn(bs, 10 if case == "scores_stride" else 5, device="cuda")
    if case == "scores_stride":
        scores = scores[:, ::2]
    kwargs = {}
    if case == "latest_stride":
        kwargs = dict(
            latest_scores=torch.randn(bs, 10, device="cuda")[:, ::2],
            request_slots=torch.arange(bs, device="cuda"),
        )
    with pytest.raises(ValueError, match="unit column stride|4096"):
        verify_layout(
            scores, query_capacity=bs * 3, **outputs(scores, bs * 3), **kwargs
        )


def test_cpu_layout_rejected():
    scores = torch.zeros(4, 5)
    with pytest.raises(ValueError, match="GPU"):
        verify_layout(scores, query_capacity=12, **outputs(scores, 12))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_padded_layout_replay_tracks_live_count_without_compiling():
    from tokenspeed_kernel.ops.attention.dsv41.triton import _verify_layout_kernel
    from utils import assert_no_triton_compile

    history = torch.randn(8, 5, device="cuda")
    latest = torch.randn(16, 5, device="cuda")
    slots = torch.arange(8, device="cuda")
    active = torch.tensor([8], dtype=torch.float32, device="cuda")
    out = outputs(history, 24)

    def run():
        verify_layout(
            history,
            query_capacity=24,
            latest_scores=latest,
            request_slots=slots,
            active_count=active,
            **out,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    # Request churn includes an idle rank. Poison padding to ensure neither
    # stale confidence nor an unknown padding row receives draft allocation.
    with assert_no_triton_compile(_verify_layout_kernel):
        for count in (8, 7, 1, 0, 4, 8):
            active.fill_(count)
            history.normal_()
            latest.normal_()
            history[count:] = float("nan")
            latest[count:8] = float("nan")
            graph.replay()
            expected = outputs(history, 24)
            reference_layout(
                history,
                24,
                **expected,
                latest_scores=latest,
                request_slots=slots,
                active_count=active,
            )
            for name in expected:
                torch.testing.assert_close(out[name], expected[name], atol=0, rtol=0)
            assert out["counts_out"][count:].tolist() == [1] * (8 - count)
    graph.reset()
