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

MODULE_PATH = Path(__file__).resolve().parents[1] / "eagle4_serving" / "eagle4_check.py"
SPEC = importlib.util.spec_from_file_location("eagle4_check", MODULE_PATH)
assert SPEC and SPEC.loader
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def test_bf16_ulp_follows_the_binade():
    assert check.bf16_ulp(16.0) == 0.125
    assert check.bf16_ulp(31.9) == 0.125
    assert check.bf16_ulp(-20.0) == 0.125
    assert check.bf16_ulp(32.0) == 0.25
    assert check.bf16_ulp(1.5) == 2.0**-7


def test_first_difference():
    assert check.first_difference([1, 2, 3], [1, 2, 3]) is None
    assert check.first_difference([1, 2, 3], [1, 2]) is None
    assert check.first_difference([1, 2, 3], [1, 4, 3]) == 1


def test_classify():
    assert check.classify([1, 2, 3], [1, 2, 3]) == ("identical", None)
    assert check.classify([1, 2, 3], [1, 2]) == ("length", None)
    assert check.classify([1, 2, 3], [1, 4, 3]) == ("diverged", 1)


def test_gap_status_sends_wide_gaps_to_fp32():
    assert check.gap_status(0.10) == "near_tie"
    assert check.gap_status(check.TIE_GAP) == "near_tie"
    assert check.gap_status(-0.125) == "near_tie"
    assert check.gap_status(0.5) == "fp32"
    assert check.gap_status(-0.25) == "fp32"


def test_fp32_status():
    # Logits near 20: one bf16 ulp is 0.125, so the tolerance is 0.25.
    logits = [0.0, 20.0, 19.8, 19.0]
    assert check.fp32_status(logits, ref=1, cand=2) == "fp32_pass"
    assert check.fp32_status(logits, ref=1, cand=3) == "fail"
    assert check.fp32_status([0.0, 20.0, 20.5], ref=1, cand=2) == "fp32_argmax"


def test_accept_lengths_reads_the_request_finish_lines():
    log = (
        "INFO Req: gnt-a Finish! Accept_num_tokens_avg: 6.2\n"
        "INFO event=ts_request_finished\n"
        "INFO Req: gnt-b Finish! Accept_num_tokens_avg: 2.857142857142857\n"
    )
    assert check.accept_lengths(log) == [6.2, 2.857142857142857]


def test_harness_serving_matches_its_measurement():
    data = Path("/data/costs")
    serve = check.serve_command("harness", 8000, 8, data, output_logprobs=False)
    measure = check.measure_command(data, 8)

    def flag(command, name):
        return command[command.index(name) + 1]

    assert flag(serve, "--speculative-exec-mode") == "0"
    assert flag(serve, "--speculative-data-path") == str(data)
    assert flag(measure, "--run-dir") == str(data)
    assert flag(measure, "--max-bs") == flag(serve, "--max-num-seqs")
    assert flag(measure, "--compute-intensity") == flag(
        serve, "--speculative-compute-intensity"
    )
    for measure_flag, serve_flag in (
        ("--topk", "--speculative-eagle-topk"),
        ("--num-steps", "--speculative-num-steps"),
        ("--num-draft-tokens", "--speculative-num-draft-tokens"),
        ("--attention-backend", "--attention-backend"),
        ("--mamba-ssm-dtype", "--mamba-ssm-dtype"),
    ):
        assert flag(measure, measure_flag) == flag(serve, serve_flag)


def test_reference_has_no_speculation_and_static_modes_no_harness():
    reference = check.serve_command(None, 8000, 8, None, output_logprobs=True)
    assert "--enable-output-logprobs" in reference
    assert "--speculative-algorithm" not in reference
    chain = check.serve_command("chain", 8000, 8, None, output_logprobs=False)
    assert "--speculative-exec-mode" not in chain
    assert chain[chain.index("--speculative-eagle-topk") + 1] == "1"
