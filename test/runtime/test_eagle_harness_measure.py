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

"""EAGLE-Harness cost measurement processing: raw samples to the cost files
serving reads (load_cost_data)."""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.execution.drafter.eagle_harness.cost import (
    MeasureRecord,
    load_cost_data,
    load_records,
    process_cache,
    process_model,
    process_query,
    robust_mean,
    save_records,
)

register_cuda_ci(est_time=2, suite="runtime-1gpu")

STEP, STOP, BS = 64, 256, 2


def test_robust_mean_drops_outliers_and_reports_convergence() -> None:
    values = [10.0] * 40 + [10.5] * 40 + [1000.0]
    mean, converged = robust_mean(values, 0.3)
    # The mean of the samples without the outlier.
    assert converged and abs(mean - 10.25) < 1e-9
    # Too spread to converge.
    _, converged = robust_mean([float(i) for i in range(1, 100)], 0.3)
    assert not converged
    # Identical samples converge on their value; all-zero samples never do.
    assert robust_mean([12.0] * 16, 0.3) == (12.0, True)
    assert not robust_mean([0.0] * 16, 0.3)[1]


def _record(us: int, count: int, q: int, kv: int, category: str):
    return MeasureRecord(us, count, q, kv, category, BS)


def test_pipeline_estimates_stage_cost_per_interval(tmp_path) -> None:
    """Attention costs 2 us per layer below 128 tokens and 3 us from there; a
    draft step costs 100 us plus its one attention layer, a verify step 1000 us
    plus 4 layers (twice the cost at width 8). The estimates add the query cost
    back to each interval's attention."""
    test, cost = tmp_path / "test" / f"bs_{BS}", tmp_path / "cost" / f"bs_{BS}"
    records = []
    for kind, widths in (("cache_slm", [1]), ("cache_llm", [1, 8])):
        for q in widths:
            # Timing jitter of +-1 us that averages out.
            records += [
                _record(
                    (2 if kv < 128 else 3) * q * 8 + (-1) ** (kv // 4), 8, q, kv, kind
                )
                for kv in range(STEP, STOP, 4)
            ]
    for kind, widths, base, layers in (
        ("model_slm", [1], 100, 1),
        ("model_llm", [1, 8], 1000, 4),
    ):
        for q in widths:
            records += [
                _record((base * q + 2 * q * layers) * 4 + (-1) ** kv, 4, q, kv, kind)
                for kv in range(64, 128)
            ]
    save_records(str(test), records)

    for kind, widths, max_q, layers in (("slm", [1], 1, 1), ("llm", [1, 8], 8, 4)):
        cache_dir, query_dir = str(cost / f"cache_{kind}"), str(cost / f"query_{kind}")
        assert process_cache(
            str(test / f"cache_{kind}"), cache_dir, widths=widths, max_query_length=max_q,
            cache_length_step=STEP, cache_length_stop=STOP, batch_size=BS, fast=True,
        ) == 0.0  # fmt: skip
        assert process_query(
            str(test / f"model_{kind}"), cache_dir, query_dir, widths=widths,
            cache_length_step=STEP, attention_layers=layers, batch_size=BS,
            loose=kind == "slm", window_layers=0, window=0,
        ) == 0.0  # fmt: skip
        process_model(
            query_dir, cache_dir, str(cost / f"model_{kind}"), widths=widths,
            max_query_length=max_q, cache_length_step=STEP, attention_layers=layers,
            batch_size=BS, window_layers=0, window=0,
        )  # fmt: skip

    with open(cost / "query_llm" / "query_8.json") as f:
        assert json.load(f)["mean"] == 8000.0
    # Intervals 0 ([64, 128)) and 1 ([128, 192)): 2 and 3 us per attention layer.
    with open(cost / "model_llm" / "model_interval_1.json") as f:
        interval = json.load(f)
    assert interval["value"] == [1000.0 + 3 * 4, 8000.0 + 24 * 4]
    assert interval["index"] == [0, 7]
    loaded = load_cost_data(str(tmp_path / "cost"))[BS]
    assert len(loaded.intervals) == (STOP - STEP) // STEP
    target, draft = loaded.intervals[0]
    assert target.value == [1008.0, 8064.0] and draft.value == [102.0]
    assert loaded.model_ratio[0] == round(1008.0 / 102.0, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_measure_policy_cycles_widths_and_lags_the_first_depth() -> None:
    from tokenspeed.runtime.execution.drafter.eagle_harness.measurement import (
        MeasurePolicy,
    )

    policy = MeasurePolicy(
        verify_num=64,
        node_top_k=8,
        tree_depth=9,
        compute_intensity="high_intensity",
        device=torch.device("cuda"),
    )
    rounds = []
    for _ in range(4):
        policy.next_widths(4, 4)
        policy.write_round()
        rounds.append((policy.widths[1:9].tolist(), policy.verify_num.item()))
    # Draft and verify widths cycle by the counter; each round's depth 1 runs the previous round's width.
    assert rounds[0] == ([8] + [2] * 7, 32)
    assert rounds[1] == ([2] + [4] * 7, 16)
    assert rounds[2] == ([4] + [8] * 7, 8)
    assert rounds[3] == ([8] + [1] * 7, 1)
    # Every depth drafts, its children after the pool's; the verify ranks 64.
    assert [policy.go_ready(d) for d in range(1, 10)] == [True] * 8 + [False]
    assert policy.cumu_starts[1:4].tolist() == [9, 73, 81]
    assert policy.cumu_count.item() == 129 and policy.verify_count.item() == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA events")
def test_measurement_counts_rounds_from_the_last_prefill_and_ends_with_eos(
    tmp_path,
) -> None:
    from tokenspeed.runtime.execution.drafter.eagle_harness.measurement import (
        FORCED_TOKEN,
        MEASURED_ROUNDS,
        CostMeasurement,
    )

    os.makedirs(tmp_path / "config")
    with open(tmp_path / "config" / "measure_config.json", "w") as f:
        json.dump({"batch_size": 3}, f)
    ranks = [
        CostMeasurement(
            run_dir=str(tmp_path), tree_depth=13, tp_group=None, file_group=None,
            rank=rank, eos_token=7, device=torch.device("cuda"),
        )  # fmt: skip
        for rank in (0, 1)
    ]

    def round_(m, bs: int, extend: bool) -> None:
        if extend:
            m.on_extend()
        else:
            m.verify_start.record()
            torch.cuda._sleep(20_000)
            m.verify_end.record()
            m.record_verify(bs)
        m.begin_draft()
        torch.cuda._sleep(20_000)
        m.end_draft(bs, 8, 64, 500)

    # A split prefill: two requests decode a round alone, then the third joins.
    schedule = [(2, True), (2, False), (3, True)] + [(3, False)] * (MEASURED_ROUNDS - 1)
    for bs, extend in schedule:
        for m in ranks:
            round_(m, bs, extend)
            assert m.commit_token.item() == FORCED_TOKEN
    # Rounds count from the batch's last prefill on every rank: the next forward commits EOS.
    for m in ranks:
        round_(m, 3, False)
        assert m.finished and m.commit_token.item() == 7
    assert ranks[0].complete() and not ranks[1].complete() and not ranks[1].records
    test_dir = ranks[0].write_model_records()
    counts = {
        category: sum(
            len(load_records(os.path.join(test_dir, category, name)))
            for name in os.listdir(os.path.join(test_dir, category))
        )
        for category in ("model_slm", "model_llm")
    }
    assert counts == {"model_slm": MEASURED_ROUNDS, "model_llm": MEASURED_ROUNDS}


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="trtllm-gen attention needs SM100",
)
def test_attention_samples_cover_the_context_grid() -> None:
    from tokenspeed.runtime.execution.drafter.eagle_harness.measure_run import (
        attention_samples,
    )

    for sliding_window in (0, 256):
        records = attention_samples(
            batch_size=2, rows=4, window=16, num_q_heads=8, num_kv_heads=2,
            head_dim=128, sliding_window=sliding_window, context_step=64,
            context_stop=1024, fast=True, repeat=2, widths=[1, 4],
            category="cache_llm",
        )  # fmt: skip
        # From one step to the stop by a sixteenth of a step, once per width.
        assert len(records) == 2 * len(range(64, 1025, 4))
        assert all(r.time_intervals > 0 and r.measure_count == 2 for r in records)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
