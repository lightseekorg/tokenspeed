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

import math

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsv41 import select_verify_budget, verify_layout


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
    verify_layout(
        scores,
        query_capacity=capacity,
        latest_scores=None,
        request_slots=None,
        active_count=None,
        **out,
    )
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
    verify_layout(
        scores,
        query_capacity=4,
        latest_scores=None,
        request_slots=None,
        active_count=None,
        **out,
    )
    assert out["valid_out"].item()
    assert out["counts_out"].tolist() == [3, 1]
    out = outputs(scores, 3)
    verify_layout(
        scores,
        query_capacity=3,
        latest_scores=None,
        request_slots=None,
        active_count=None,
        **out,
    )
    assert not out["valid_out"].item()
    assert out["counts_out"].tolist() == [1, 1]
    assert out["rows_out"].tolist() == [0, 3, 0]
    scores.fill_(-1000)
    out = outputs(scores, 4)
    verify_layout(
        scores,
        query_capacity=4,
        latest_scores=None,
        request_slots=None,
        active_count=None,
        **out,
    )
    assert out["counts_out"].tolist() == [2, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_graph_replay_reads_changed_scores_and_keeps_output_storage():
    scores = torch.zeros((4, 5), device="cuda")
    out = outputs(scores, 12)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            verify_layout(
                scores,
                query_capacity=12,
                latest_scores=None,
                request_slots=None,
                active_count=None,
                **out,
            )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        verify_layout(
            scores,
            query_capacity=12,
            latest_scores=None,
            request_slots=None,
            active_count=None,
            **out,
        )
    for row in range(4):
        scores.fill_(-10)
        scores[row].fill_(10)
        graph.replay()
        expected = outputs(scores, 12)
        verify_layout(
            scores,
            query_capacity=12,
            latest_scores=None,
            request_slots=None,
            active_count=None,
            **expected,
        )
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
            verify_layout(
                scores,
                query_capacity=capacity,
                latest_scores=None,
                request_slots=None,
                active_count=None,
                **actual,
            )
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
            history,
            query_capacity=24,
            latest_scores=latest,
            request_slots=slots,
            active_count=None,
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
            active_count=None,
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
    kwargs.setdefault("latest_scores", None)
    kwargs.setdefault("request_slots", None)
    kwargs.setdefault("active_count", None)
    with pytest.raises(ValueError, match="unit column stride|4096"):
        verify_layout(
            scores, query_capacity=bs * 3, **outputs(scores, bs * 3), **kwargs
        )


def test_cpu_layout_rejected():
    scores = torch.zeros(4, 5)
    with pytest.raises(ValueError, match="GPU"):
        verify_layout(
            scores,
            query_capacity=12,
            latest_scores=None,
            request_slots=None,
            active_count=None,
            **outputs(scores, 12),
        )


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


def _reference_select_verify_budget(
    latest_scores: torch.Tensor,
    request_slots: torch.Tensor,
    seq_lens: torch.Tensor,
    profiles: list[tuple[int, int, float, dict[int, float]]],
    capacities: list[int],
    *,
    padded_bs: int,
    context_advance: int,
    min_batch_size: int,
    min_gain: float,
    policy_ms: float,
) -> tuple[int, int, int]:
    bs = int(request_slots.shape[0])
    draft = int(latest_scores.shape[1])
    full = padded_bs * (draft + 1)
    seq_vals = [int(v) for v in seq_lens.tolist()]
    min_ctx = min(seq_vals) if seq_vals else 0
    max_ctx = max(seq_vals) if seq_vals else 0
    if bs < min_batch_size or bs <= 0 or any(v <= 0 for v in seq_vals):
        return full, min_ctx, max_ctx
    rows = []
    for slot in request_slots.tolist():
        if slot < 0 or slot >= latest_scores.shape[0]:
            return full, min_ctx, max_ctx
        row = latest_scores[slot].tolist()
        if any(not math.isfinite(v) for v in row):
            return full, min_ctx, max_ctx
        rows.append(row)
    upper_ctx = max_ctx + context_advance
    chosen = None
    for p_min, p_max, native_ms, step_map in profiles:
        if p_min <= upper_ctx <= p_max:
            chosen = (native_ms, step_map)
            break
    if chosen is None or not math.isfinite(chosen[0]) or chosen[0] <= 0:
        return full, min_ctx, max_ctx
    native_ms, step_map = chosen
    gains = []
    for row in rows:
        survival = 1.0
        for score in row:
            exp = math.exp(-abs(score))
            survival *= 1.0 / (1.0 + exp) if score >= 0.0 else exp / (1.0 + exp)
            gains.append(survival)
    gains.sort(reverse=True)
    prefix = [0.0]
    for g in gains:
        prefix.append(prefix[-1] + g)
    full_reward = float(bs) + prefix[len(gains)]
    best_q = full
    best_rate = (full_reward / native_ms) * (1.0 + min_gain)
    for q in sorted(capacities, reverse=True):
        if q == full or q < padded_bs or q >= full:
            continue
        step_ms = step_map.get(q)
        if step_ms is None or not math.isfinite(step_ms) or step_ms <= 0:
            continue
        reward = float(bs) + prefix[min(max(q - padded_bs, 0), len(gains))]
        rate = reward / (step_ms + policy_ms)
        if rate > best_rate:
            best_q, best_rate = q, rate
    return best_q, min_ctx, max_ctx


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_select_verify_budget_matches_policy_and_edge_cases():
    draft = 5
    slots_total = 32
    torch.manual_seed(7)
    latest = torch.randn(slots_total, draft, dtype=torch.float32, device="cuda")
    # Distinct score rows so selection tests both packing and context boundaries.
    latest[:4] = 8.0
    latest[4:8] = -8.0

    capacities = [36, 28, 20, 12]
    profiles = [
        (1, 8192, 10.0, {36: 8.0, 28: 6.0, 20: 4.0, 12: 2.5}),
        (8193, 32768, 12.0, {36: 9.0, 28: 8.5, 20: 9.5}),
    ]
    policy_ms = 0.25
    prof_ranges = torch.tensor(
        [[p[0], p[1]] for p in profiles], dtype=torch.int32, device="cuda"
    )
    prof_native = torch.tensor(
        [p[2] for p in profiles], dtype=torch.float32, device="cuda"
    )
    prof_caps = torch.tensor(capacities, dtype=torch.int32, device="cuda")
    prof_step = torch.tensor(
        [
            [p[3][c] + policy_ms if c in p[3] else -1.0 for c in capacities]
            for p in profiles
        ],
        dtype=torch.float32,
        device="cuda",
    )
    out = torch.empty(3, dtype=torch.int32, device="cuda")

    def check(slot_list, ctx_list, *, padded_bs=8, adv=12, min_bs=2, min_gain=0.02):
        # Trailing padded slots hold non-finite poison that must be ignored past bs.
        all_slots = torch.tensor(
            slot_list + [slots_total - 1] * (padded_bs - len(slot_list)),
            dtype=torch.int32,
            device="cuda",
        )[: len(slot_list)]
        seq_lens = torch.tensor(ctx_list, dtype=torch.int32, device="cuda")
        select_verify_budget(
            latest,
            request_slots=all_slots,
            seq_lens=seq_lens,
            profile_ranges=prof_ranges,
            profile_native_ms=prof_native,
            profile_capacities=prof_caps,
            profile_step_ms=prof_step,
            selected_out=out,
            padded_bs=padded_bs,
            context_advance=adv,
            min_batch_size=min_bs,
            min_gain=min_gain,
        )
        expected = _reference_select_verify_budget(
            latest,
            all_slots,
            seq_lens,
            profiles,
            capacities,
            padded_bs=padded_bs,
            context_advance=adv,
            min_batch_size=min_bs,
            min_gain=min_gain,
            policy_ms=policy_ms,
        )
        assert tuple(out.tolist()) == expected

    # Poison tail row not referenced by active slice; ensures padded_bs > bs skips tail.
    latest[slots_total - 1] = float("nan")
    check(list(range(8)), [4000] * 8)
    check([0, 1, 4, 5, 6], [100, 200, 300, 8180, 128])
    check([0, 1, 4, 5, 6], [100, 200, 300, 8181, 128])
    # Uncalibrated upper bound -> native full width (8 * 6 = 48).
    check([0, 1, 4, 5], [100, 200, 300, 32760])
    # Below min_batch_size or zero seq_len -> native full width.
    check([0], [500], min_bs=2)
    check([0, 1], [500, 0])
    # Empty batch -> full width, [0, 0] bounds.
    check([], [])
    # Non-finite active slot -> native full width.
    latest[2, 3] = float("inf")
    check([0, 1, 2, 3], [500] * 4)

    for empty_ranges, empty_native, empty_caps, empty_step in (
        (prof_ranges[:0], prof_native[:0], prof_caps, prof_step[:0]),
        (prof_ranges, prof_native, prof_caps[:0], prof_step[:, :0]),
    ):
        with pytest.raises(ValueError, match="non-empty"):
            select_verify_budget(
                latest,
                request_slots=torch.arange(4, dtype=torch.int32, device="cuda"),
                seq_lens=torch.full((4,), 100, dtype=torch.int32, device="cuda"),
                profile_ranges=empty_ranges,
                profile_native_ms=empty_native,
                profile_capacities=empty_caps,
                profile_step_ms=empty_step,
                selected_out=out,
                padded_bs=4,
                context_advance=8,
                min_batch_size=1,
                min_gain=0.02,
            )

    # Tie on rate selects largest capacity (prof_caps descending order).
    tie_latest = torch.full((4, 3), -60.0, dtype=torch.float32, device="cuda")
    tie_ranges = torch.tensor([[1, 1000]], dtype=torch.int32, device="cuda")
    tie_native = torch.tensor([100.0], dtype=torch.float32, device="cuda")
    tie_caps = torch.tensor([12, 8], dtype=torch.int32, device="cuda")
    tie_step = torch.tensor([[10.0, 10.0]], dtype=torch.float32, device="cuda")
    select_verify_budget(
        tie_latest,
        request_slots=torch.arange(4, dtype=torch.int32, device="cuda"),
        seq_lens=torch.full((4,), 100, dtype=torch.int32, device="cuda"),
        profile_ranges=tie_ranges,
        profile_native_ms=tie_native,
        profile_capacities=tie_caps,
        profile_step_ms=tie_step,
        selected_out=out,
        padded_bs=4,
        context_advance=8,
        min_batch_size=1,
        min_gain=0.02,
    )
    assert out.tolist() == [12, 100, 100]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_select_verify_budget_runtime_shapes_do_not_compile():
    from tokenspeed_kernel.ops.attention.dsv41.triton import (
        _select_verify_budget_kernel,
    )
    from utils import assert_no_triton_compile

    draft = 5
    latest_full = torch.randn(64, draft, dtype=torch.float32, device="cuda")
    ranges_full = torch.tensor(
        [[1, 4096], [4097, 16384], [16385, 65536]], dtype=torch.int32, device="cuda"
    )
    native_full = torch.tensor([10.0, 12.0, 15.0], dtype=torch.float32, device="cuda")
    caps_full = torch.tensor([64, 48, 32, 24], dtype=torch.int32, device="cuda")
    step_full = torch.tensor(
        [
            [6.0, 5.0, 4.0, 3.0],
            [7.0, 6.0, 5.0, 4.0],
            [11.0, 10.0, 9.0, 8.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    out = torch.empty(3, dtype=torch.int32, device="cuda")
    slots = torch.arange(16, dtype=torch.int32, device="cuda")
    seq_lens = torch.full((16,), 1024, dtype=torch.int32, device="cuda")

    select_verify_budget(
        latest_full,
        request_slots=slots,
        seq_lens=seq_lens,
        profile_ranges=ranges_full,
        profile_native_ms=native_full,
        profile_capacities=caps_full,
        profile_step_ms=step_full,
        selected_out=out,
        padded_bs=16,
        context_advance=12,
        min_batch_size=1,
        min_gain=0.02,
    )
    with assert_no_triton_compile(_select_verify_budget_kernel):
        for bs in (1, 5, 11, 16):
            for n_prof in (1, 2, 3):
                for n_cap in (1, 2, 4):
                    select_verify_budget(
                        latest_full[:32] if bs <= 5 else latest_full,
                        request_slots=slots[:bs],
                        seq_lens=seq_lens[:bs],
                        profile_ranges=ranges_full[:n_prof],
                        profile_native_ms=native_full[:n_prof],
                        profile_capacities=caps_full[:n_cap],
                        profile_step_ms=step_full[:n_prof, :n_cap],
                        selected_out=out,
                        padded_bs=16,
                        context_advance=6,
                        min_batch_size=2,
                        min_gain=0.05,
                    )
        assert out[0].item() > 0
