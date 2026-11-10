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

import importlib.util
from pathlib import Path

from pipeline import check_perf_reference, extract_perf_summary_rows

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "single_request_benchmark" / "gsm8k_decode.py"
)
SPEC = importlib.util.spec_from_file_location("gsm8k_decode", MODULE_PATH)
assert SPEC and SPEC.loader
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def test_last_boxed_takes_the_final_answer_with_nested_braces():
    text = "try \\boxed{1}, then \\boxed{\\frac{3}{4}} done"
    assert bench.last_boxed(text) == "\\frac{3}{4}"
    assert bench.last_boxed("no answer") is None
    assert bench.last_boxed("cut off \\boxed{12") is None


def test_answers_compare_as_numbers():
    reference = "She sells 16 - 3 - 4 = <<16-3-4=9>>9 eggs.\n#### 1,018"
    assert bench.is_correct("so \\boxed{1018}", reference)
    assert bench.is_correct("so \\boxed{\\$1,018.00}", reference)
    assert bench.is_correct("so \\boxed{1018 \\text{ dollars}}", reference)
    assert not bench.is_correct("so \\boxed{1017}", reference)
    assert not bench.is_correct("so \\boxed{}", reference)
    assert not bench.is_correct("1018", reference)


def test_decode_rate_leaves_out_the_first_token():
    # 101 tokens: the first at 1.0 s, the last at 1.5 s -> 100 tokens in 0.5 s.
    result = bench.RequestResult(101, 1.0, 1.5, "")
    assert result.decode_tok_s == 200.0
    assert bench.RequestResult(1, 1.0, 1.0, "").decode_tok_s is None


def test_summary_csv_passes_the_pipeline_perf_gate():
    results = [
        bench.RequestResult(101, 0.0, 0.5, ""),  # 200 tok/s
        bench.RequestResult(41, 0.0, 0.1, ""),  # 400 tok/s
    ]
    csv = bench.summary_csv("cfg", results, wall_s=1.0, num_gpus=2)
    rows = extract_perf_summary_rows(f"log line\n{csv}\nmore log")
    assert rows == [
        {
            "config": "cfg",
            "Conc.": "1",
            "Latency (tps/user)": "300.00",
            "Throughput (tps/gpu)": "71.00",
            "Approx Cache Hit": "N/A",
            "Decoded Tok/Iter": "N/A",
        }
    ]
    results_by_stage = [{"stage": "perf", "perf_summary_rows": rows}]
    task = {"perf_reference": {1: [320, 70]}, "perf_threshold": 0.9}
    assert check_perf_reference(task, results_by_stage, ["perf"])["passed"]
    task = {"perf_reference": {1: [400, 70]}, "perf_threshold": 0.9}
    assert not check_perf_reference(task, results_by_stage, ["perf"])["passed"]
