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
import scheduler_perf_guard as guard

ANNOTATE_OUTPUT = """\
--------------------------------------------------------------------------------
Ir                      file:function
--------------------------------------------------------------------------------
825,617,708 (100.0%)  ???:main [/bench]
  285,005,619 (34.52%)  ???:tokenspeed::CacheCoordinator::Admit(tokenspeed::CacheCoordinator::PrefixProbe&&) [/bench]
    1,945,340 ( 0.24%)  ???:std::deque<X>::emplace_back<tokenspeed::CacheCoordinator::Admit(int)::{lambda()#1}> [/bench]
       12,000 ( 0.00%)  /src/cache_admission.cpp:tokenspeed::CacheCoordinator::AdmitLater() [/bench]
"""


def test_reads_callgrind_totals():
    assert (
        guard.parse_callgrind_totals("events: Ir\nsummary: 1234\ntotals: 1234\n")
        == 1234
    )
    with pytest.raises(guard.GuardError):
        guard.parse_callgrind_totals("events: Ir\n")


def test_inclusive_cost_counts_only_the_function_itself():
    assert (
        guard.parse_inclusive_cost(ANNOTATE_OUTPUT, guard.ADMIT_FUNCTION) == 285_005_619
    )
    with pytest.raises(guard.GuardError):
        guard.parse_inclusive_cost(ANNOTATE_OUTPUT, "tokenspeed::Scheduler::Advance(")


def test_per_sample_cost_cancels_setup():
    # 1000 instructions of setup, 50 per sample.
    assert guard.per_sample(1000 + 10 * 50, 1000 + 40 * 50, 10, 40) == 50
    with pytest.raises(guard.GuardError):
        guard.per_sample(1, 2, 3, 3)


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (102.5, "regression"),
        (101.5, "within_budget"),
        (98.5, "within_budget"),
        (97.0, "improvement"),
    ],
)
def test_instruction_budget(candidate, expected):
    assert guard.classify_instructions(100.0, candidate)[0] == expected


def _results(
    scale_admit: float, decode_admit_64k: float
) -> dict[str, dict[str, float]]:
    results = {
        case.name: {"api": 1000.0, "admit": 100.0} for case in guard.INSTRUCTION_CASES
    }
    results["full_cache_1m_hit0"]["api"] = 1000.0 * scale_admit
    results["decode_64k"]["admit"] = decode_admit_64k
    return results


def _status(report, check):
    return next(item["status"] for item in report["scaling"] if item["check"] == check)


def test_scaling_checks_block_new_growth_only():
    # Decode admission growing with context blocks when the base did not grow.
    report = guard.instruction_report(_results(1.0, 100.0), _results(1.0, 224.0))
    assert report["status"] == "regression"
    assert _status(report, "decode_admit_vs_context") == "regression"
    # Growth the base already had is reported without blocking...
    report = guard.instruction_report(_results(1.0, 224.0), _results(1.0, 224.0))
    assert _status(report, "decode_admit_vs_context") == "preexisting"
    assert report["status"] == "passed"
    # ...and fixing it passes.
    report = guard.instruction_report(_results(1.0, 224.0), _results(1.0, 100.5))
    assert _status(report, "decode_admit_vs_context") == "passed"
    assert _status(report, "admission_vs_cache_size") == "passed"


def test_a_base_without_a_benchmark_bootstraps_its_cases():
    base = _results(1.0, 100.0)
    for name in ("decode_8k", "decode_64k"):
        del base[name]
    report = guard.instruction_report(base, _results(1.0, 100.0))
    assert report["status"] == "bootstrap"
    decode = [
        item for item in report["comparisons"] if item["case"].startswith("decode")
    ]
    assert {item["classification"] for item in decode} == {"bootstrap"}
    assert _status(report, "decode_admit_vs_context") == "passed"
    # A candidate regression still blocks while other cases bootstrap.
    report = guard.instruction_report(base, _results(1.0, 200.0))
    assert report["status"] == "regression"
    assert "within_budget" in guard.render_instructions(report)


def test_timing_uses_the_median_round():
    # One slow round does not decide; two of three do.
    assert (
        guard.classify_timing([(0.30, 3.0), (0.01, 0.1), (0.00, 0.0)])[0]
        == "within_budget"
    )
    assert (
        guard.classify_timing([(0.30, 3.0), (0.20, 2.0), (0.00, 0.0)])[0]
        == "regression"
    )
    # A large relative change of a tiny time stays within budget.
    assert (
        guard.classify_timing([(0.50, 0.01), (0.50, 0.01), (0.50, 0.01)])[0]
        == "within_budget"
    )
    # Rounds on opposite sides of the budget void the verdict.
    assert (
        guard.classify_timing([(0.30, 3.0), (-0.20, -2.0), (0.10, 1.0)])[0]
        == "inconclusive"
    )
    assert (
        guard.classify_timing([(-0.10, -1.0), (-0.20, -2.0), (0.00, 0.0)])[0]
        == "improvement"
    )


def test_round_change_compares_medians():
    change, delta = guard.round_change([1.0, 2.0, 100.0], [2.0, 4.0, 4.0])
    assert change == pytest.approx(1.0)
    assert delta == pytest.approx(2.0)


def test_a_candidate_without_benchmarks_is_an_error(tmp_path):
    exit_code = guard.main(
        [
            "instructions",
            "--candidate",
            str(tmp_path / "missing"),
            "--base",
            "",
            "--output",
            str(tmp_path / "out"),
            "--valgrind",
            "valgrind",
            "--callgrind-annotate",
            "callgrind_annotate",
        ]
    )
    assert exit_code == 2
